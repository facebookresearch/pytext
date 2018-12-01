#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

from pytext.config import ConfigBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import Model
from pytext.models.output_layers import ClassificationOutputLayer
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.pure_doc_attention import PureDocAttention


class DocModel(Model):
    """
    An n-ary document classification model.
    """

    class Config(ConfigBase):
        representation: Union[
            PureDocAttention.Config,
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config,
        ] = BiLSTMDocAttention.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )
