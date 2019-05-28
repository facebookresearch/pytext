#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
from pytext.models.doc_model import DocModel_Deprecated

from .ensemble import Ensemble


class BaggingDocEnsemble_Deprecated(Ensemble):
    """Ensemble class that uses bagging for ensembling document classification
    models.
    """

    class Config(Ensemble.Config):
        """Configuration class for `BaggingDocEnsemble_Deprecated`.
        These attributes are used by `Ensemble.from_config()` to construct
        instance of `BaggingDocEnsemble_Deprecated`.

        Attributes:
            models (List[DocModel_Deprecated.Config]): List of document classification model
                configurations.

        """

        models: List[DocModel_Deprecated.Config]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Call `forward()` method of each document classification sub-model by
        passing all arguments and named arguments to the sub-models, collect the
        logits from them and average their values.

        Returns:
            torch.Tensor: Logits from the ensemble.

        """
        logit_d_list = torch.cat(
            tuple(model.forward(*args, **kwargs).unsqueeze(2) for model in self.models),
            dim=2,
        )

        return torch.mean(logit_d_list, dim=2)
