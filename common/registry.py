#!/usr/bin/env python3
import collections
from typing import Any, Dict, Iterable, Type

import pytext.optimizers as opt


JOB_SPEC = "job_spec"
DATA_HANDLER = "data_handler"
TRAINER = "trainer"
LOSS = "loss"
OPTIMIZER = "optimizer"
MODEL = "model"
PREDICTOR = "predictor"
EXPORTER = "exporter"


class RegistryError(Exception):
    pass


class Registry:
    _registered_components: Dict[str, Dict[Type, Type]] = collections.defaultdict(dict)

    @classmethod
    def add(cls, component_type: str, cls_to_add: Type, config_cls: Type):
        component = cls._registered_components[component_type]
        if config_cls in component:
            raise RegistryError(
                f"Cannot add {cls_to_add} to {component_type} for task_config type \
                {config_cls}; it's already registered for \
                {component[config_cls]}"
            )
        component[config_cls] = cls_to_add

    @classmethod
    def get(cls, component_type: str, config_cls: Type) -> Type:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        if config_cls not in cls._registered_components[component_type]:
            raise RegistryError(
                f"unregistered config class {config_cls} for {component_type} "
            )
        return cls._registered_components[component_type][config_cls]

    @classmethod
    def values(cls, component_type: str) -> Iterable[Type]:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        return cls._registered_components[component_type].values()


# TODO T32608581 create a base component class with from_config method
def _from_config(cls, config, **metadata):
    return cls(config, **metadata)


def jobspec(cls: Type):
    Registry.add(JOB_SPEC, cls, cls)
    return cls


def component(component_type: str, config_cls: Type):
    """Decorator to register a class
    """

    def register(cls: Type):
        if not hasattr(cls, "from_config"):
            cls.from_config = classmethod(_from_config)
        Registry.add(component_type, cls, config_cls)
        return cls

    return register


def create_component(component_type: str, config: Any, **kwargs):
    config_cls = type(config)
    cls = Registry.get(component_type, config_cls)
    return cls.from_config(config, **kwargs)


def create_data_handler(data_handler_config, feature_config, label_config, **kwargs):
    return create_component(
        DATA_HANDLER,
        data_handler_config,
        feature_config=feature_config,
        label_config=label_config,
        **kwargs,
    )


def create_trainer(trainer_config, **kwargs):
    return create_component(TRAINER, trainer_config, **kwargs)


def create_model(model_config, feature_config, **kwargs):
    return create_component(MODEL, model_config, feat_config=feature_config, **kwargs)


def create_predictor(predictor_config, **kwargs):
    return create_component(PREDICTOR, predictor_config, **kwargs)


def create_exporter(exporter_config, feature_config, label_config, **kwargs):
    return create_component(
        EXPORTER,
        exporter_config,
        feature_config=feature_config,
        label_config=label_config,
        **kwargs,
    )


def create_loss(loss_config, **kwargs):
    return create_component(LOSS, loss_config, **kwargs)


# TODO think about how to refactor optimizer creator
def create_optimizer(optimizer_config, model, **kwargs):
    return opt.create_optimizer(model, optimizer_config)
