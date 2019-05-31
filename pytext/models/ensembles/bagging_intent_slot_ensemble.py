#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
from pytext.models.joint_model import IntentSlotModel, JointModel
from pytext.models.model import Model
from pytext.models.output_layers import CRFOutputLayer

from .ensemble import Ensemble_Deprecated, EnsembleModel


class BaggingIntentSlotEnsemble_Deprecated(Ensemble_Deprecated):
    """Ensemble class that uses bagging for ensembling intent-slot models.

    Args:
        config (Config): Configuration object specifying all the
            parameters of BaggingIntentSlotEnsemble_Deprecated.
        models (List[Model]): List of intent-slot model objects.

    Attributes:
        use_crf (bool): Whether to use CRF for word tagging task.
        output_layer (IntentSlotOutputLayer): Output layer of intent-slot
            model responsible for computing loss and predictions.

    """

    class Config(Ensemble_Deprecated.Config):
        """Configuration class for `BaggingIntentSlotEnsemble_Deprecated`.
        These attributes are used by `Ensemble.from_config()` to construct
        instance of `BaggingIntentSlotEnsemble_Deprecated`.

        Attributes:
            models (List[JointModel.Config]): List of intent-slot model configurations.
            output_layer (IntentSlotOutputLayer): Output layer of intent-slot
                model responsible for computing loss and predictions.

        """

        models: List[JointModel.Config]
        use_crf: bool = False

    def __init__(self, config: Config, models: List[Model], *args, **kwargs) -> None:
        super().__init__(config, models)
        self.use_crf = isinstance(self.output_layer.word_output, CRFOutputLayer)

    def merge_sub_models(self) -> None:
        """Merges all sub-models' transition matrices when using CRF.
        Otherwise does nothing.
        """
        # to get the transition_matrix for the ensemble model, we average the
        # transition matrices of the children model
        if not self.use_crf:
            return
        transition_matrix = torch.mean(
            torch.cat(
                tuple(
                    model.output_layer.word_output.crf.get_transitions().unsqueeze(0)
                    for model in self.models
                ),
                dim=0,
            ),
            dim=0,
        )
        self.output_layer.word_output.crf.set_transitions(transition_matrix)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Call `forward()` method of each intent-slot sub-model by passing all
        arguments and named arguments to the sub-models, collect the logits from
        them and average their values.

        Returns:
            torch.Tensor: Logits from the ensemble.

        """
        logit_d_list, logit_w_list = None, None
        for model in self.models:
            logit_d, logit_w = model.forward(*args, **kwargs)
            logit_d, logit_w = logit_d.unsqueeze(2), logit_w.unsqueeze(3)

            if logit_d_list is None:
                logit_d_list = logit_d
            else:
                logit_d_list = torch.cat([logit_d_list, logit_d], dim=2)

            if logit_w_list is None:
                logit_w_list = logit_w
            else:
                logit_w_list = torch.cat([logit_w_list, logit_w], dim=3)

        return [torch.mean(logit_d_list, dim=2), torch.mean(logit_w_list, dim=3)]


class BaggingIntentSlotEnsembleModel(EnsembleModel):
    """Ensemble class that uses bagging for ensembling intent-slot models.

    Args:
        config (Config): Configuration object specifying all the
            parameters of BaggingIntentSlotEnsemble.
        models (List[Model]): List of intent-slot model objects.

    Attributes:
        use_crf (bool): Whether to use CRF for word tagging task.
        output_layer (IntentSlotOutputLayer): Output layer of intent-slot
            model responsible for computing loss and predictions.

    """

    class Config(EnsembleModel.Config):
        """Configuration class for `BaggingIntentSlotEnsemble`.
        These attributes are used by `Ensemble.from_config()` to construct
        instance of `BaggingIntentSlotEnsemble`.

        Attributes:
            models (List[IntentSlotModel.Config]): List of intent-slot model
                configurations.
            output_layer (IntentSlotOutputLayer): Output layer of intent-slot
                model responsible for computing loss and predictions.

        """

        models: List[IntentSlotModel.Config]
        use_crf: bool = False

    def __init__(self, config: Config, models: List[Model], *args, **kwargs) -> None:
        super().__init__(config, models)
        self.use_crf = isinstance(self.output_layer.word_output, CRFOutputLayer)

    def merge_sub_models(self) -> None:
        """Merges all sub-models' transition matrices when using CRF.
        Otherwise does nothing.
        """
        # to get the transition_matrix for the ensemble model, we average the
        # transition matrices of the children model
        if not self.use_crf:
            return
        transition_matrix = torch.mean(
            torch.cat(
                tuple(
                    model.output_layer.word_output.crf.get_transitions().unsqueeze(0)
                    for model in self.models
                ),
                dim=0,
            ),
            dim=0,
        )
        self.output_layer.word_output.crf.set_transitions(transition_matrix)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Call `forward()` method of each intent-slot sub-model by passing all
        arguments and named arguments to the sub-models, collect the logits from
        them and average their values.

        Returns:
            torch.Tensor: Logits from the ensemble.

        """
        logit_d_list, logit_w_list = None, None
        for model in self.models:
            logit_d, logit_w = model.forward(*args, **kwargs)
            logit_d, logit_w = logit_d.unsqueeze(2), logit_w.unsqueeze(3)

            if logit_d_list is None:
                logit_d_list = logit_d
            else:
                logit_d_list = torch.cat([logit_d_list, logit_d], dim=2)

            if logit_w_list is None:
                logit_w_list = logit_w
            else:
                logit_w_list = torch.cat([logit_w_list, logit_w], dim=3)

        return [torch.mean(logit_d_list, dim=2), torch.mean(logit_w_list, dim=3)]
