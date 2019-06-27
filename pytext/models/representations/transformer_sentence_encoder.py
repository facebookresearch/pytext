#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import torch
from fairseq.modules import (
    TransformerSentenceEncoder as TransformerSentenceEncoderModule,
)
from pytext.config import ConfigBase
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)


class TransformerSentenceEncoder(TransformerSentenceEncoderBase):
    """
    Implementation of the Transformer Sentence Encoder. This directly makes
    use of the TransformerSentenceEncoder module in Fairseq.

    A few interesting config options:
        - encoder_normalize_before detemines whether the layer norm is applied
          before or after self_attention. This is similar to original
          implementation from Google.
        - activation_fn can be set to 'gelu' instead of the default of 'relu'.
        - project_representation adds a linear projection + tanh to the pooled output
          in the style of BERT.
    """

    class Config(TransformerSentenceEncoderBase.Config, ConfigBase):
        # Dropout parameters
        dropout: float = 0.1
        attention_dropout: float = 0.1
        activation_dropout: float = 0.1

        # Parameters related to hidden states and self-attention
        embedding_dim: int = 768
        ffn_embedding_dim: int = 3072
        num_encoder_layers: int = 6
        num_attention_heads: int = 8
        num_segments: int = 2

        # Parameters related to positions
        use_position_embeddings: bool = True
        # the fairseq module for position embeddings offsets all position
        # ids by the padding index. Disable this offset by setting this flag
        # to False. This will work correctly since we mask out the embeddings
        # associated with padding in the encoder
        offset_positions_by_padding: bool = True

        # Model Initialization parameters
        apply_bert_init: bool = True

        # Misc. Params
        encoder_normalize_before: bool = True
        activation_fn: str = "relu"
        project_representation: bool = False
        max_seq_len: int = 128

        # multilingual is set to true for cross-lingual LM training
        multilingual: bool = False

        # Flags for freezing parameters (e.g. during fine-tuning)
        freeze_embeddings: bool = False
        n_trans_layers_to_freeze: int = 0

    def __init__(
        self,
        config: Config,
        output_encoded_layers: bool,
        padding_idx: int,
        vocab_size: int,
        *args,
        **kwarg,
    ) -> None:

        super().__init__(config, output_encoded_layers=output_encoded_layers)
        self.multilingual = config.multilingual
        self.offset_positions_by_padding = config.offset_positions_by_padding

        self.sentence_encoder = TransformerSentenceEncoderModule(
            padding_idx=padding_idx,
            vocab_size=vocab_size,
            num_encoder_layers=config.num_encoder_layers,
            embedding_dim=config.embedding_dim,
            ffn_embedding_dim=config.ffn_embedding_dim,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            num_segments=config.num_segments,
            use_position_embeddings=config.use_position_embeddings,
            offset_positions_by_padding=config.offset_positions_by_padding,
            encoder_normalize_before=config.encoder_normalize_before,
            apply_bert_init=config.apply_bert_init,
            activation_fn=config.activation_fn,
            freeze_embeddings=config.freeze_embeddings,
            n_trans_layers_to_freeze=config.n_trans_layers_to_freeze,
            export=self.export,
        )
        self.projection = (
            torch.nn.Linear(self.representation_dim, self.representation_dim)
            if config.project_representation
            else None
        )

    def _encoder(
        self, input_tuple: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        # If multilingual is True then the input_tuple has additional information
        # related to the lengths of the inputs as well as a position tensor
        if self.multilingual:
            tokens, _, lengths, segment_labels, positions = input_tuple

            # we need this for backwards compatibility with models that are
            # pre-trained with the offset
            if self.offset_positions_by_padding:
                positions = None
        else:
            tokens, _, segment_labels = input_tuple
            positions = None

        encoded_layers, pooled_output = self.sentence_encoder(
            tokens, segment_labels, positions=positions
        )
        # Each tensor in encoded_layers output by the Fairseq module has
        # the shape: T x B x C. Convert this to B x T x C
        encoded_layers = [x.transpose(0, 1) for x in encoded_layers]
        if self.projection:
            pooled_output = self.projection(pooled_output).tanh()
        return encoded_layers, pooled_output

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.sentence_encoder.embed_tokens
