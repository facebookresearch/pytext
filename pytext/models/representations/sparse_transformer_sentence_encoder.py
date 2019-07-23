#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fairseq.modules.sparse_transformer_sentence_encoder import (
    SparseTransformerSentenceEncoder as SparseTransformerSentenceEncoderModule,
)
from pytext.config import ConfigBase
from pytext.models.representations.transformer_sentence_encoder import (
    TransformerSentenceEncoder,
)


class SparseTransformerSentenceEncoder(TransformerSentenceEncoder):
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

    class Config(TransformerSentenceEncoder.Config, ConfigBase):
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

        # Sparse multihead attention parameters
        is_bidirectional: bool = True
        stride: int = 32
        expressivity: int = 8

    def __init__(
        self,
        config: Config,
        output_encoded_layers: bool,
        padding_idx: int,
        vocab_size: int,
        *args,
        **kwarg,
    ) -> None:

        super().__init__(
            config,
            output_encoded_layers=output_encoded_layers,
            padding_idx=padding_idx,
            vocab_size=vocab_size,
        )
        self.sentence_encoder = SparseTransformerSentenceEncoderModule(
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
            is_bidirectional=config.is_bidirectional,
            stride=config.stride,
            expressivity=config.expressivity,
        )
