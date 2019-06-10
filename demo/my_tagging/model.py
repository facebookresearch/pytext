#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
from pytext.config.pytext_config import ConfigBase
from pytext.data.tensorizers import TokenTensorizer
from pytext.loss import CrossEntropyLoss
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention

from .output import MyTaggingOutputLayer


class MyTaggingModel(Model):
    class Config(ConfigBase):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            slots: TokenTensorizer.Config = TokenTensorizer.Config(column="slots")

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: BiLSTMSlotAttention.Config = BiLSTMSlotAttention.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: MyTaggingOutputLayer.Config = MyTaggingOutputLayer.Config()

    @classmethod
    def from_config(cls, config, tensorizers):
        embedding = create_module(config.embedding, tensorizer=tensorizers["tokens"])
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        slots = tensorizers["slots"].vocab
        decoder = create_module(
            config.decoder, in_dim=representation.representation_dim, out_dim=len(slots)
        )
        output_layer = MyTaggingOutputLayer(slots, CrossEntropyLoss(None))
        return cls(embedding, representation, decoder, output_layer)

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        slots, _, _ = tensor_dict["slots"]
        return slots

    def forward(
        self, word_tokens: torch.Tensor, seq_lens: torch.Tensor
    ) -> List[torch.Tensor]:
        embedding = self.embedding(word_tokens)
        representation = self.representation(embedding, seq_lens)

        # some LSTM representations return extra tensors
        if isinstance(representation, tuple):
            representation = representation[0]

        return self.decoder(representation)
