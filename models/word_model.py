#!/usr/bin/env python3

from typing import Union

from pytext.config import ConfigBase
from pytext.models.model import Model
from pytext.models.output_layer import CRFOutputLayer, WordTaggingOutputLayer
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention
from pytext.models.representations.biseqcnn import BSeqCNNRepresentation


class WordTaggingModel(Model):
    """
    Word tagging model.
    """

    class Config(ConfigBase):
        representation: Union[BiLSTMSlotAttention.Config, BSeqCNNRepresentation.Config]
        output_config: Union[WordTaggingOutputLayer.Config, CRFOutputLayer.Config]
        proj_config: MLPProjection.Config = MLPProjection.Config()
