#!/usr/bin/env python3

from pytext.config import ConfigBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import Model
from pytext.models.output_layer import ClassificationOutputLayer
from pytext.models.representations.seq_rep import SeqRepresentation


class SeqNNModel(Model):
    """
    Sequence of sequences labeling model: for example, dialog or long text.
    it uses a docnn model (CNN or LSTM) to generate vector representation
    for each sequence, and then use an LSTM or BLSTM to capture the dynamics
    and produce labels for each sequence.
    Reference: in deeptext 2.0
    https://our.intern.facebook.com/intern/wiki/Using_DeepText_sequence_(SeqNN)_models/
    SeqNNc data in Hive tables:
        a column named text (of type ARRAY<STRING>) - a sequence of strings
        a column named label (of type STRING) - the true class/label of that sequence

    """

    class Config(ConfigBase):
        representation: SeqRepresentation.Config = SeqRepresentation.Config()
        output_layer: ClassificationOutputLayer.Config = ClassificationOutputLayer.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
