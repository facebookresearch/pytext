#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Any, Dict, List

import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.component import create_model
from pytext.config.field_config import FeatureConfig
from pytext.data.tensorizers import Tensorizer
from pytext.models.model import Model


class Ensemble_Deprecated(Model):
    """Base class for ensemble models.

    Args:
        config (Config): Configuration object specifying all the parameters of
            Ensemble.
        models (List[Model]): List of sub-model objects.

    Attributes:
        output_layer (OutputLayerBase): Responsible for computing loss and predictions.
        models (nn.ModuleList]): ModuleList container for sub-model objects.

    """

    class Config(ConfigBase):
        models: List[Any]
        sample_rate: float = 1.0

    @classmethod
    def from_config(cls, config: Config, feat_config: FeatureConfig, *args, **kwargs):
        """Factory method to construct an instance of Ensemble or one its derived
        classes from the module's config object and the field's metadata object.
        It creates sub-models in the ensemble using the sub-model's configuration.

        Args:
            config (Config): Configuration object specifying all the
                parameters of Ensemble.
            feat_config (FeatureConfig): Configuration object specifying all the
                parameters of the input features to the model.

        Returns:
            type: An instance of Ensemble.

        """
        sub_models = [
            create_model(sub_model_config, feat_config, *args, **kwargs)
            for sub_model_config in config.models
        ]
        return cls(config, sub_models, *args, **kwargs)

    def __init__(self, config: Config, models: List[Model], *args, **kwargs) -> None:
        nn.Module.__init__(self)
        self.models = nn.ModuleList(models)
        self.output_layer = deepcopy(models[0].output_layer)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def merge_sub_models(self):
        pass

    def save_modules(self, base_path: str = "", suffix: str = "") -> None:
        """Saves the modules of all sub_models in the `Ensemble`.

        Args:
            base_path (str): Path of base directory. Defaults to "".
            suffix (str): Suffix to add to the file name to save. Defaults to "".

        """
        for model in self.models:
            model.save_modules(base_path, suffix)


class EnsembleModel(Ensemble_Deprecated):
    __EXPANSIBLE__ = True

    class Config(Ensemble_Deprecated.Config):
        @property
        def inputs(self):
            return self.models[0].inputs

    @classmethod
    def from_config(
        cls, config: Config, tensorizers: Dict[str, Tensorizer], *args, **kwargs
    ):
        """Factory method to construct an instance of Ensemble or one its derived
        classes from the module's config object and tensorizers
        It creates sub-models in the ensemble using the sub-model's configuration.

        Args:
            config (Config): Configuration object specifying all the
                parameters of Ensemble.
            tensorizers (Dict[str, Tensorizer]): Tensorizer specifying all the
                parameters of the input features to the model.

        Returns:
            type: An instance of Ensemble.

        """
        sub_models = [
            create_model(sub_model_config, tensorizers, *args, **kwargs)
            for sub_model_config in config.models
        ]
        return cls(config, sub_models, *args, **kwargs)

    def arrange_model_inputs(self, tensor_dict):
        return self.models[0].arrange_model_inputs(tensor_dict)

    def arrange_targets(self, tensor_dict):
        return self.models[0].arrange_targets(tensor_dict)

    def arrange_model_context(self, tensor_dict):
        return self.models[0].arrange_model_context(tensor_dict)
