#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional

from pytext.config import ConfigBase
from pytext.data.bert_tensorizer import BERTTensorizer
from pytext.data.tensorizers import NumericLabelTensorizer, Tensorizer
from pytext.models.bert_classification_models import BertPairwiseModel, NewBertModel
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.module import create_module
from pytext.models.output_layers import (
    PairwiseCosineRegressionOutputLayer,
    RegressionOutputLayer,
)
from pytext.utils.usage import log_class_usage


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

    def __init__(self, encoder, decoder, output_layer) -> None:
        super().__init__(encoder, decoder, output_layer)
        log_class_usage(__class__)


class BertPairwiseRegressionModel(BertPairwiseModel):
    """
    Two-tower model for regression. Encode two texts separately and use the cosine
    similarity between sentence embeddings to predict regression label.
    """

    class Config(BertPairwiseModel.Config):
        class ModelInput(BertPairwiseModel.Config.ModelInput):
            labels: NumericLabelTensorizer.Config = NumericLabelTensorizer.Config()

        inputs: ModelInput = ModelInput()
        decoder: Optional[MLPDecoder.Config] = None
        output_layer: PairwiseCosineRegressionOutputLayer.Config = (
            PairwiseCosineRegressionOutputLayer.Config()
        )
        encode_relations: bool = False
