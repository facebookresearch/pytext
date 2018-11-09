#!/usr/bin/env python3
import collections
import enum
from typing import Any, Dict, Iterable, List, Tuple, Type, Union

import torch

from .pytext_config import ConfigBase, PyTextConfig


class ComponentType(enum.Enum):
    JOB_SPEC = "job_spec"
    DATA_HANDLER = "data_handler"
    FEATURIZER = "featurizer"
    TRAINER = "trainer"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    MODEL = "model"
    MODULE = "module"
    PREDICTOR = "predictor"
    EXPORTER = "exporter"
    METRIC_REPORTER = "metric_reporter"


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

                class Config(parent_config, ConfigBase):
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


class Component(metaclass=ComponentMeta):
    class Config(ConfigBase):
        pass

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(config, *args, **kwargs)

    def __init__(self, config=None):
        self.config = config


def register_jobspec(cls_list: Union[Type, List[Type]]):
    if not isinstance(cls_list, Iterable):
        cls_list = [cls_list]
    for cls in cls_list:
        Registry.add(ComponentType.JOB_SPEC, cls, cls)
    PyTextConfig._field_types["jobspec"].__args__ = tuple(
        Registry.values(ComponentType.JOB_SPEC)
    )


def create_component(component_type: ComponentType, config: Any, *args, **kwargs):
    config_cls = type(config)
    cls = Registry.get(component_type, config_cls)
    return cls.from_config(config, *args, **kwargs)


def create_data_handler(data_handler_config, *args, **kwargs):
    return create_component(
        ComponentType.DATA_HANDLER, data_handler_config, *args, **kwargs
    )


def create_featurizer(featurizer_config, *args, **kwargs):
    return create_component(
        ComponentType.FEATURIZER, featurizer_config, *args, **kwargs
    )


def create_trainer(trainer_config, *args, **kwargs):
    return create_component(ComponentType.TRAINER, trainer_config, *args, **kwargs)


def create_model(model_config, *args, **kwargs):
    return create_component(ComponentType.MODEL, model_config, *args, **kwargs)


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
