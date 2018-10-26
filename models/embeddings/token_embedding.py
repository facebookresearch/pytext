#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytext.common.constants import DatasetFieldName
from pytext.config.field_config import FeatureConfig
from pytext.data import CommonMetadata
from pytext.models.module import Module

from .char_embedding import CharacterEmbedding
from .dict_embedding import DictEmbedding
from .pretrained_model_embedding import PretrainedModelEmbedding


class TokenEmbedding(Module):
    """
    Single point of entry for embedding a token in all possible ways.
    It encapsulates all things that should be used for token representation.

    It takes care of
    1. word level embedding
    2. character level embedding
    3. dictionary feature embedding
    4. <add yours>

    Add all token embedding logic to this class.
    Please DO NOT add independent embeddings in your model.
    """

    Config = FeatureConfig

    # TODO @kushall @zsc we need to think more about the design here.
    # 1 how to support more embedding types
    # 2 are embeddings mapping to features 1:1?
    # 3 shall each embedding class also register themself and do it's own from_config?
    # 4 shall we get metedata in a non-flattened way?
    @classmethod
    def from_config(cls, config: FeatureConfig, metadata: CommonMetadata):
        word_embed, dict_embed, char_embed, pretrained_model_embed = cls.get_embed(
            config, metadata
        )
        return cls(config, word_embed, dict_embed, char_embed, pretrained_model_embed)

    @classmethod
    def get_embed(cls, config: FeatureConfig, metadata: CommonMetadata):
        word_embed = None
        dict_embed = None
        char_embed = None
        pretrained_model_embed = None

        if config.word_feat.embed_dim:
            word_feat_meta = metadata.features[DatasetFieldName.TEXT_FIELD]
            word_embed = nn.Embedding(
                word_feat_meta.vocab_size,
                config.word_feat.embed_dim,
                _weight=metadata.pretrained_embeds_weight,
                sparse=config.word_feat.sparse,
            )
            embedding_init_range = config.word_feat.embedding_init_range
            if (
                metadata.pretrained_embeds_weight is None
                and embedding_init_range is not None
            ):
                word_embed.weight.data.uniform_(
                    embedding_init_range[0], embedding_init_range[1]
                )
            # Initialize unk embedding with zeros
            # to guard the model against randomized decisions based on unknown words
            word_embed.weight.data[word_feat_meta.unk_token_idx].fill_(0.0)
            word_embed.weight.requires_grad = not config.word_feat.freeze

        if config.dict_feat:
            dict_feat_meta = metadata.features[DatasetFieldName.DICT_FIELD]
            dict_embed = DictEmbedding(
                dict_feat_meta.vocab_size,
                config.dict_feat.embed_dim,
                config.dict_feat.pooling,
                sparse=config.dict_feat.sparse,
            )
        if config.char_feat:
            char_feat_meta = metadata.features[DatasetFieldName.CHAR_FIELD]
            char_embed = CharacterEmbedding(
                char_feat_meta.vocab_size,
                config.char_feat.embed_dim,
                config.char_feat.cnn.kernel_num,
                config.char_feat.cnn.kernel_sizes,
                sparse=config.char_feat.sparse,
            )
        if config.pretrained_model_embedding:
            pretrained_model_embed = PretrainedModelEmbedding(
                config.pretrained_model_embedding.embed_dim
            )
        return word_embed, dict_embed, char_embed, pretrained_model_embed

    def __init__(
        self,
        config: FeatureConfig,
        word_embed: Optional[nn.Embedding],
        dict_embed: Optional[DictEmbedding],
        char_embed: Optional[CharacterEmbedding],
        pretrained_model_embed: Optional[PretrainedModelEmbedding],
    ) -> None:
        super().__init__(config)
        self.embedding_dim = 0
        self.word_embed = word_embed
        self.dict_embed = dict_embed
        self.char_embed = char_embed
        self.pretrained_model_embed = pretrained_model_embed
        if word_embed:
            self.embedding_dim += word_embed.embedding_dim

        if dict_embed:
            self.embedding_dim += dict_embed.embedding_dim

        if char_embed:
            self.embedding_dim += char_embed.embedding_dim

        if pretrained_model_embed:
            self.embedding_dim += pretrained_model_embed.embedding_dim

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor = None,
        dict_feat: Tuple[torch.Tensor, ...] = None,
        chars: torch.Tensor = None,
        pretrained_model_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        # tokens dim: (bsz, max_seq_len) -> (bsz, max_seq_len, dim)
        # or (bsz, max_num_sen, max_seq_len) -> (bsz, max_num_sen, max_seq_len, dim)
        # for seqnn
        embeddings = []

        if self.word_embed:
            word_emb = self.word_embed(tokens)
            embeddings.append(word_emb)

        if self.dict_embed:
            if not dict_feat:
                raise ValueError("dict_feat argument is None.")
            dict_ids, dict_weights, dict_lengths = dict_feat
            dict_emb = self.dict_embed(dict_ids, dict_weights, dict_lengths)
            embeddings.append(dict_emb)

        if self.char_embed:
            if chars is None:
                raise ValueError("char_feat argument is None")
            char_emb = self.char_embed(chars)
            embeddings.append(char_emb)

        if self.pretrained_model_embed:
            if pretrained_model_embedding is None:
                raise ValueError("pretrained_model_embedding argument is None")
            if (
                pretrained_model_embedding.shape[2]
                != self.pretrained_model_embed.embedding_dim
            ):
                raise ValueError(
                    "Expected {} as dimension for pretrained_model_embedding but {} received".format(
                        self.pretrained_model_embed.embedding_dim,
                        pretrained_model_embedding.shape[2],
                    )
                )
            embeddings.append(pretrained_model_embedding)
        return torch.cat(embeddings, 2)
