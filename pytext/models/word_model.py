#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import Model
from pytext.models.output_layers import CRFOutputLayer, WordTaggingOutputLayer
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention
from pytext.models.representations.biseqcnn import BSeqCNNRepresentation


class WordTaggingModel(Model):
    """
    Word tagging model. It can be used for any task that requires predicting the
    tag for a word/token. For example, the following tasks can be modeled as word
    tagging tasks. This is not an exhaustive list.
    1. Part of speech tagging.
    2. Named entity recognition.
    3. Slot filling for task oriented dialog.

    It can be instantiated just like any other :class:`~Model`.
    """

    class Config(Model.Config):
        representation: Union[
            BiLSTMSlotAttention.Config, BSeqCNNRepresentation.Config
        ] = BiLSTMSlotAttention.Config()
        output_layer: Union[
            WordTaggingOutputLayer.Config, CRFOutputLayer.Config
        ] = WordTaggingOutputLayer.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
