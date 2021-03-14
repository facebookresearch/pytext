#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional, Dict, Union, Tuple

from pytext.config import ConfigBase
from pytext.models.module import create_module
from torch import Tensor

from .base import PyTextSeq2SeqModule
from .conv_decoder import LightConvDecoder, LightConvDecoupledDecoder
from .conv_encoder import LightConvEncoder


class CNNModel(PyTextSeq2SeqModule):
    class Config(ConfigBase):
        encoder: LightConvEncoder.Config = LightConvEncoder.Config()
        decoder: Union[
            LightConvDecoder.Config, LightConvDecoupledDecoder.Config
        ] = LightConvDecoder.Config()

    @classmethod
    def from_config(
        cls,
        config: Config,
        src_dict,
        source_embedding,
        tgt_dict,
        target_embedding,
        dict_embedding=None,
    ):
        cls.validate_config(config)
        encoder = create_module(config.encoder, src_dict, source_embedding)
        decoder = create_module(config.decoder, tgt_dict, target_embedding)
        return cls(encoder, decoder, source_embedding)

    def __init__(self, encoder, decoder, source_embedding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embeddings = source_embedding

    def forward(
        self,
        src_tokens: Tensor,
        additional_features: List[List[Tensor]],
        src_lengths,
        prev_output_tokens,
        src_subword_begin_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # embed tokens
        embeddings = self.source_embeddings([[src_tokens]] + additional_features)

        encoder_out = self.encoder(src_tokens, embeddings, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

    def max_decoder_positions(self):
        return max(self.encoder.max_positions(), self.decoder.max_positions())

    def get_embedding_module(self):
        return self.source_embeddings

    @classmethod
    def validate_config(cls, config):
        assert (
            config.encoder.encoder_config.max_target_positions
            <= config.decoder.decoder_config.max_target_positions
        )


class DecoupledCNNModel(CNNModel):
    class Config(CNNModel.Config):
        encoder: LightConvEncoder.Config = LightConvEncoder.Config()
        decoder: LightConvDecoupledDecoder.Config = LightConvDecoupledDecoder.Config()
