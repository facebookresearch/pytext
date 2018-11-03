#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytext.common.constants import DatasetFieldName
from pytext.config.field_config import FeatureConfig
from pytext.data import CommonMetadata

from .char_embedding import CharacterEmbedding
from .dict_embedding import DictEmbedding
from .pretrained_model_embedding import PretrainedModelEmbedding
from .token_embedding import TokenEmbedding


class SequenceTokenEmbedding(TokenEmbedding):
    """
    sentence level embedding
    """

    @classmethod
    def from_config(cls, config: FeatureConfig, metadata: CommonMetadata):
        word_embed, dict_embed, char_embed, pretrained_model_embed = super().get_embed(
            config, metadata
        )
        if config.seq_word_feat:
            seq_word_feat_meta = metadata.features[DatasetFieldName.SEQ_FIELD]
            seq_word_embed = nn.Embedding(
                seq_word_feat_meta.vocab_size,
                config.seq_word_feat.embed_dim,
                _weight=seq_word_feat_meta.pretrained_embeds_weight,
                sparse=config.seq_word_feat.sparse,
            )
            embedding_init_range = config.seq_word_feat.embedding_init_range
            if (
                seq_word_feat_meta.pretrained_embeds_weight is None
                and embedding_init_range is not None
            ):
                seq_word_embed.weight.data.uniform_(
                    embedding_init_range[0], embedding_init_range[1]
                )
            # Initialize unk embedding with zeros
            # to guard the model against randomized decisions based on unknown words
            seq_word_embed.weight.data[seq_word_feat_meta.unk_token_idx].fill_(0.0)
            seq_word_embed.weight.requires_grad = not config.seq_word_feat.freeze
        return cls(
            config,
            word_embed,
            dict_embed,
            char_embed,
            pretrained_model_embed,
            seq_word_embed,
        )

    def __init__(
        self,
        config: FeatureConfig,
        word_embed: Optional[nn.Embedding],
        dict_embed: Optional[DictEmbedding],
        char_embed: Optional[CharacterEmbedding],
        pretrained_model_embed: Optional[PretrainedModelEmbedding],
        seq_word_embed: Optional[nn.Embedding],
    ) -> None:
        super().__init__(
            config, word_embed, dict_embed, char_embed, pretrained_model_embed
        )
        self.seq_word_embedding_dim = 0
        self.seq_word_embed = seq_word_embed
        if seq_word_embed:
            self.seq_word_embedding_dim = seq_word_embed.embedding_dim

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor = None,
        dict_feat: Tuple[torch.Tensor, ...] = None,
        chars: torch.Tensor = None,
        pretrained_model_embedding: torch.Tensor = None,
        seq_tokens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, ...]:
        token_embed = super().forward(
            tokens, seq_lens, dict_feat, chars, pretrained_model_embedding
        )
        if self.seq_word_embed is None:
            raise ValueError("seq_feat argument is unspecified.")
        seq_word_embed = self.seq_word_embed(seq_tokens)
        return (token_embed, seq_word_embed)
