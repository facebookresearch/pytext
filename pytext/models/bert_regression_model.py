#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

from pytext.config import ConfigBase
from pytext.data.bert_tensorizer import BERTTensorizer
from pytext.data.tensorizers import NumericLabelTensorizer, Tensorizer
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.module import create_module
from pytext.models.output_layers import RegressionOutputLayer


class NewBertRegressionModel(NewBertModel):
    """BERT single sentence (or concatenated sentences) regression."""

    class Config(NewBertModel.Config):
        class InputConfig(ConfigBase):
            tokens: BERTTensorizer.Config = BERTTensorizer.Config(
                columns=["text1", "text2"], max_seq_len=128
            )
            labels: NumericLabelTensorizer.Config = NumericLabelTensorizer.Config()

        inputs: InputConfig = InputConfig()
        output_layer: RegressionOutputLayer.Config = RegressionOutputLayer.Config()

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        vocab = tensorizers["tokens"].vocab
        encoder = create_module(
            config.encoder,
            padding_idx=vocab.get_pad_index(),
            vocab_size=vocab.__len__(),
        )
        decoder = create_module(
            config.decoder, in_dim=encoder.representation_dim, out_dim=1
        )
        output_layer = RegressionOutputLayer.from_config(config.output_layer)
        return cls(encoder, decoder, output_layer)
