#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import collections
import enum
from typing import Any, Dict, List, Tuple, Type, Union

import torch

from .pytext_config import ConfigBase, PyTextConfig


class ComponentType(enum.Enum):
    TASK = "task"
    COLUMN = "column"
    DATA_TYPE = "data_type"
    DATA_HANDLER = "data_handler"
    DATA_SOURCE = "data_source"
    TOKENIZER = "tokenizer"
    TENSORIZER = "tensorizer"
    BATCHER = "batcher"
    BATCH_SAMPLER = "batch_sampler"
    FEATURIZER = "featurizer"
    TRAINER = "trainer"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    MODEL = "model"
    MODEL2 = "model2"
    MODULE = "module"
    PREDICTOR = "predictor"
    EXPORTER = "exporter"
    METRIC_REPORTER = "metric_reporter"
    SPARSIFIER = "sparsifier"
    MASKING_FUNCTION = "masking_function"
    PRIVACY_ENGINE = "privacy_engine"


class RegistryError(Exception):
    pass


class Registry:
    _registered_components: Dict[ComponentType, Dict[Type, Type]] = (
        collections.defaultdict(dict)
    )

    @classmethod
    def add(cls, component_type: ComponentType, cls_to_add: Type, config_cls: Type):
        component = cls._registered_components[component_type]
        if config_cls in component:
            raise RegistryError(
                f"Cannot add {cls_to_add} to {component_type} "
                f"for task_config type {config_cls}; "
                f"it's already registered for {component[config_cls]}"
            )
        component[config_cls] = cls_to_add

    @classmethod
    def get(cls, component_type: ComponentType, config_cls: Type) -> Type:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        if config_cls not in cls._registered_components[component_type]:
            raise RegistryError(
                f"unregistered config class {config_cls.__name__} for {component_type}"
            )
        return cls._registered_components[component_type][config_cls]

    @classmethod
    def values(cls, component_type: ComponentType) -> Tuple[Type, ...]:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        return tuple(cls._registered_components[component_type].values())

    @classmethod
    def configs(cls, component_type: ComponentType) -> Tuple[Type, ...]:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        return tuple(cls._registered_components[component_type].keys())

    @classmethod
    def subconfigs(cls, config_cls: Type) -> Tuple[Type, ...]:
        return tuple(
            sub_cls
            for sub_cls in cls.configs(config_cls.__COMPONENT_TYPE__)
            if issubclass(sub_cls.__COMPONENT__, config_cls.__COMPONENT__)
        )


class ComponentMeta(type):
    def __new__(metacls, typename, bases, namespace):
        if "Config" not in namespace:
            # We need to dynamically create a new Config class per
            # instance rather than inheriting a single empty config class
            # because components are registered uniquely by config class.
            # If a parent class specifies a config class, inherit from it.
            parent_config = next(
                (base.Config for base in bases if hasattr(base, "Config")), None
            )
            if parent_config is not None:

                class Config(parent_config):
                    pass

            else:

                class Config(ConfigBase):
                    pass

            namespace["Config"] = Config

        component_type = next(
            (
                base.__COMPONENT_TYPE__
                for base in bases
                if hasattr(base, "__COMPONENT_TYPE__")
            ),
            namespace.get("__COMPONENT_TYPE__"),
        )
        new_cls = super().__new__(metacls, typename, bases, namespace)

        new_cls.Config.__COMPONENT_TYPE__ = component_type
        new_cls.Config.__name__ = f"{typename}.Config"
        new_cls.Config.__COMPONENT__ = new_cls
        new_cls.Config.__EXPANSIBLE__ = namespace.get("__EXPANSIBLE__")
        if component_type:
            Registry.add(component_type, new_cls, new_cls.Config)
        return new_cls

    def __dir__(cls):
        """Jit doesnt allow scripting of attributes whose classname includes "."

        Example Repro:

        class OldModule(Module):
            class Config(ConfigBase):
                a: int = 5

            @classmethod
            def from_config(cls, config: Config):
                return cls(config.a)

            def __init__(self, a):
                super().__init__()
                self.a = a

            def forward(self, b: int) -> int:
                return b + self.a

        m = OldModule.from_config(OldModule.Config())
        jit.script(m)

            > RuntimeError: Could not get qualified name for class 'OldModule.Config':
        'OldModule.Config' is not a valid identifier

        print(m.Config.__name__)
            > OldModule.Config

        At the sametime, we dont need to script the config classes because they
        are not needed during inference time. Hence in this workaround we skip
        the config classes.

        Ideal solution is that when building models they should be inheriting
        from nn.Module only and not Component. This requires significant changes
        to the way models are created in PyText.

        """
        result = super().__dir__()
        return [
            r
            for r in result
            if not (
                isinstance(getattr(cls, r, None), type)
                and issubclass(getattr(cls, r, None), ConfigBase)
            )
        ]


class Component(metaclass=ComponentMeta):
    class Config(ConfigBase):
        pass

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(config, *args, **kwargs)

    def __init__(self, config=None, *args, **kwargs):
        self.config = config


def register_tasks(task_cls: Union[Type, List[Type]]):
    """
        Task classes are already added to registry during declaration, pass them
        as parameters here just to make sure they're imported
    """
    vars(PyTextConfig)["__annotations__"]["task"].__args__ = Registry.configs(
        ComponentType.TASK
    )


def create_component(component_type: ComponentType, config: Any, *args, **kwargs):
    config_cls = type(config)
    cls = Registry.get(component_type, config_cls)
    try:
        return cls.from_config(config, *args, **kwargs)
    except TypeError as e:
        raise Exception(f"Can't create component {cls}: {str(e)}")


def create_data_handler(data_handler_config, *args, **kwargs):
    return create_component(
        ComponentType.DATA_HANDLER, data_handler_config, *args, **kwargs
    )


def create_featurizer(featurizer_config, *args, **kwargs):
    return create_component(
        ComponentType.FEATURIZER, featurizer_config, *args, **kwargs
    )


def create_trainer(trainer_config, model: torch.nn.Module, *args, **kwargs):
    return create_component(
        ComponentType.TRAINER, trainer_config, model, *args, **kwargs
    )


def create_model(model_config, *args, **kwargs):
    return create_component(ComponentType.MODEL, model_config, *args, **kwargs)


def create_optimizer(optimizer_config, model: torch.nn.Module, *args, **kwargs):
    return create_component(
        ComponentType.OPTIMIZER, optimizer_config, model, *args, **kwargs
    )


def create_scheduler(scheduler_config, optimizer, *args, **kwargs):
    if hasattr(optimizer, "fp32_optimizer"):
        optimizer = optimizer.fp32_optimizer
    return create_component(
        ComponentType.SCHEDULER, scheduler_config, optimizer, *args, **kwargs
    )


def create_sparsifier(sparsifier_config, *args, **kwargs):
    return create_component(
        ComponentType.SPARSIFIER, sparsifier_config, *args, **kwargs
    )


def create_privacy_engine(privacy_engine_config, *args, **kwargs):
    return create_component(
        ComponentType.PRIVACY_ENGINE, privacy_engine_config, *args, **kwargs
    )


def create_predictor(predictor_config, *args, **kwargs):
    return create_component(ComponentType.PREDICTOR, predictor_config, *args, **kwargs)


def create_exporter(exporter_config, *args, **kwargs):
    return create_component(ComponentType.EXPORTER, exporter_config, *args, **kwargs)


def create_loss(loss_config, *args, **kwargs):
    return create_component(ComponentType.LOSS, loss_config, *args, **kwargs)


def create_metric_reporter(module_config, *args, **kwargs):
    return create_component(
        ComponentType.METRIC_REPORTER, module_config, *args, **kwargs
    )


def get_component_name(obj):
    """
        Return the human-readable name of the class of `obj`.
        Document the type of a config field and can be used as a Union value
        in a json config.
    """
    if obj is type(None):
        return None
    if not hasattr(obj, "__module__"):  # builtins
        return obj.__class__.__name__
    if obj.__module__ == "typing":
        return str(obj)[7:]
    if hasattr(obj, "__qualname__"):  # Class name unaltered by meta
        ret = obj.__qualname__
    elif hasattr(obj, "__name__"):
        ret = obj.__name__
    else:
        ret = obj.__class__.__name__
    if ret.endswith(".Config"):
        return obj.__COMPONENT__.__name__
    return ret
