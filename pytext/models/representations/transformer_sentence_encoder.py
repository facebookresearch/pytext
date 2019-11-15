#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import torch
from fairseq.modules import (
    TransformerSentenceEncoder as TransformerSentenceEncoderModule,
)
from pytext.config import ConfigBase
from pytext.models.representations.traced_transformer_encoder import (
    TracedTransformerEncoder,
)
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
        - projection_dim adds a linear projection to projection_dim + tanh to
          the pooled output in the style of BERT.
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
        projection_dim: int = 0
        max_seq_len: int = 128

        # multilingual is set to true for cross-lingual LM training
        multilingual: bool = False

        # Flags for freezing parameters (e.g. during fine-tuning)
        freeze_embeddings: bool = False
        n_trans_layers_to_freeze: int = 0

        # Use of TorchScript and optimizations
        use_torchscript: bool = False

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
        self.use_torchscript = config.use_torchscript
        self.traced_encoder = None

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
        if self.use_torchscript:
            assert hasattr(self.sentence_encoder, "traceable")
            self.sentence_encoder.traceable = self.use_torchscript

        self.projection = (
            torch.nn.Linear(self.representation_dim, config.projection_dim)
            if config.projection_dim > 0
            else None
        )

    def load_state_dict(self, state_dict):
        self.upgrade_state_dict_named(state_dict)
        # "projection" must be be in sync with the name of member variable projection.
        has_projection = any("projection" in key for key in state_dict.keys())
        if self.projection is not None and not has_projection:
            projection_temp = self.projection
            self.projection = None
            super().load_state_dict(state_dict)
            self.projection = projection_temp
        else:
            super().load_state_dict(state_dict)

    def _encoder(
        self, input_tuple: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        tokens, _, segment_labels, positions = input_tuple
        if self.offset_positions_by_padding or (not self.multilingual):
            positions = None

        if self.use_torchscript and self.traced_encoder is None:
            self.traced_encoder = TracedTransformerEncoder(
                self.sentence_encoder, tokens, segment_labels, positions
            )
            del self.sentence_encoder
            self.sentence_encoder = self.traced_encoder
            print("Using traced transformer sentence encoder")

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

    def upgrade_state_dict_named(self, state_dict):
        # We convert in_proj_weight to individual q,k,v weights
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith("in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[k.replace("in_proj_weight", "q_proj.weight")] = state_dict[
                    k
                ][:dim]
                items_to_add[k.replace("in_proj_weight", "k_proj.weight")] = state_dict[
                    k
                ][dim : 2 * dim]
                items_to_add[k.replace("in_proj_weight", "v_proj.weight")] = state_dict[
                    k
                ][2 * dim :]
                keys_to_remove.append(k)

            if k.endswith("in_proj_bias"):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[k.replace("in_proj_bias", "q_proj.bias")] = state_dict[k][
                    :dim
                ]
                items_to_add[k.replace("in_proj_bias", "k_proj.bias")] = state_dict[k][
                    dim : 2 * dim
                ]
                items_to_add[k.replace("in_proj_bias", "v_proj.bias")] = state_dict[k][
                    2 * dim :
                ]
                keys_to_remove.append(k)

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

        return state_dict
