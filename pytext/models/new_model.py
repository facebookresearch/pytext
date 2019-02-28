#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase, ConfigBaseMeta
from pytext.data.tensorizers import Tensorizer


class ModelInputMeta(ConfigBaseMeta):
    def __new__(metacls, typename, bases, namespace):
        annotations = namespace.get("__annotations__", {})
        for type in annotations.values():
            if not issubclass(type, Tensorizer.Config):
                raise TypeError(
                    "ModelInput configuration should only include tensorizers"
                )
        return super().__new__(metacls, typename, bases, namespace)


class ModelInputBase(ConfigBase, metaclass=ModelInputMeta):
    """Base class for model inputs."""


class Model(Component):
    __COMPONENT_TYPE__ = ComponentType.MODEL2
    __EXPANSIBLE__ = True

    class Config(Component.Config):
        class ModelInput(ModelInputBase):
            pass

        inputs: ModelInput = ModelInput()

    def train_batch(self, batch):
        model_inputs = self.arrange_model_inputs(batch)
        model_outputs = self(*model_inputs)
        loss = self.get_loss(model_outputs, self.arrange_targets(batch), None)
        predictions, scores = self.get_pred(model_outputs)
        targets = self.arrange_targets(batch)
        # These are another reason I think it might make sense for model
        # to own metric reporting
        metric_data = (predictions, targets, scores, loss, model_inputs)
        return loss, metric_data

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        raise NotImplementedError
