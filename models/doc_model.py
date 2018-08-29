#!/usr/bin/env python3

from typing import Union

from pytext.config import ConfigBase
from pytext.models.model import Model
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_self_attn import BiLSTMSelfAttention
from pytext.models.representations.docnn import DocNNRepresentation


class DocModel(Model):
    """
    An n-ary document classification model.
    """
    class Config(ConfigBase):
        representation: Union[BiLSTMSelfAttention.Config, DocNNRepresentation.Config]
        proj_config: MLPProjection.Config = MLPProjection.Config()
