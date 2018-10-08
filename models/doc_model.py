#!/usr/bin/env python3

from typing import Union

from pytext.config import ConfigBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import Model
from pytext.models.output_layer import ClassificationOutputLayer
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation


class DocModel(Model):
    """
    An n-ary document classification model.
    """

    class Config(ConfigBase):
        representation: Union[
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config,
        ] = BiLSTMDocAttention.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )
        decoder: MLPDecoder.Config = MLPDecoder.Config()
