#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional

import torch.jit
from pytext.config import ConfigBase
from pytext.models.module import create_module
from pytext.utils.usage import log_class_usage

from .base import PyTextSeq2SeqModule
from .rnn_decoder import RNNDecoder
from .rnn_encoder import LSTMSequenceEncoder


class RNNModel(PyTextSeq2SeqModule):
    class Config(ConfigBase):
        encoder: LSTMSequenceEncoder.Config = LSTMSequenceEncoder.Config()
        decoder: RNNDecoder.Config = RNNDecoder.Config()

    def __init__(self, encoder, decoder, source_embeddings):
        super().__init__()
        self.source_embeddings = source_embeddings
        self.encoder = encoder
        self.decoder = decoder
        log_class_usage(__class__)

    def forward(
        self,
        src_tokens: torch.Tensor,
        additional_features: List[List[torch.Tensor]],
        src_lengths,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # embed tokens
        embeddings = self.source_embeddings([[src_tokens]] + additional_features)

        # n.b. tensorized_features[0][0] must be src_tokens
        encoder_out = self.encoder(src_tokens, embeddings, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, incremental_state)
        return decoder_out

    @classmethod
    def from_config(
        cls,
        config: Config,
        source_vocab,
        source_embedding,
        target_vocab,
        target_embedding,
    ):
        out_vocab_size = len(target_vocab)
        encoder = create_module(config.encoder)
        decoder = create_module(config.decoder, out_vocab_size, target_embedding)
        return cls(encoder, decoder, source_embedding)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

    def max_decoder_positions(self):
        return max(self.encoder.max_positions(), self.decoder.max_positions())
