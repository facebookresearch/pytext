#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.config import ConfigBase
from pytext.data.tensorizers import SeqTokenTensorizer
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.doc_model import DocModel
from pytext.models.model import Model
from pytext.models.output_layers import ClassificationOutputLayer
from pytext.models.representations.seq_rep import SeqRepresentation


class SeqNNModel_Deprecated(Model):
    """
    Classification model with sequence of utterances as input.
    It uses a docnn model (CNN or LSTM) to generate vector representation
    for each sequence, and then use an LSTM or BLSTM to capture the dynamics
    and produce labels for each sequence.

    DEPRECATED: Use SeqNNModel
    """

    class Config(ConfigBase):
        representation: SeqRepresentation.Config = SeqRepresentation.Config()
        output_layer: ClassificationOutputLayer.Config = ClassificationOutputLayer.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()


class SeqNNModel(DocModel):
    """
    Classification model with sequence of utterances as input.
    It uses a docnn model (CNN or LSTM) to generate vector representation
    for each sequence, and then use an LSTM or BLSTM to capture the dynamics
    and produce labels for each sequence.
    """

    class Config(DocModel.Config):
        class ModelInput(DocModel.Config.ModelInput):
            tokens: SeqTokenTensorizer.Config = SeqTokenTensorizer.Config(
                column="text_seq"
            )

        inputs: ModelInput = ModelInput()
        representation: SeqRepresentation.Config = SeqRepresentation.Config()

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens = tensor_dict["tokens"]
        model_inputs = (tokens, seq_lens)
        if "dense" in tensor_dict:
            model_inputs += (tensor_dict["dense"],)
        return model_inputs
