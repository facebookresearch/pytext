#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.config import ConfigBase
from pytext.data.bert_tensorizer import RoBERTaTensorizer
from pytext.data.tensorizers import LabelTensorizer
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.module import Module, create_module
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)


class RoBERTaEncoder(TransformerSentenceEncoderBase):
    class Config(TransformerSentenceEncoderBase.Config):
        pretrained_encoder: Module.Config = Module.Config(
            load_path=(
                "manifold://pytext_training/tree/static/models/roberta_public.pt1"
            )
        )

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        self.encoder = create_module(config.pretrained_encoder)
        self.representation_dim = self.encoder.encoder.token_embedding.weight.size(-1)

    def _encoder(self, inputs):
        # NewBertModel expects the output as a tuple and grabs the first element
        tokens, _, _ = inputs
        full_representation = self.encoder(tokens)
        sentence_rep = full_representation[:, 0, :]
        return [full_representation], sentence_rep


class RoBERTa(NewBertModel):
    class Config(NewBertModel.Config):
        class InputConfig(ConfigBase):
            tokens: RoBERTaTensorizer.Config = RoBERTaTensorizer.Config()
            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: InputConfig = InputConfig()
        encoder: RoBERTaEncoder.Config = RoBERTaEncoder.Config()
