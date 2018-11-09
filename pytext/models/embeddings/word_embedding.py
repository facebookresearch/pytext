#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
from pytext.config.field_config import WordFeatConfig
from pytext.fields import FieldMeta

from .embedding_base import EmbeddingBase


class WordEmbedding(EmbeddingBase, nn.Embedding):
    Config = WordFeatConfig

    @classmethod
    def from_config(cls, config: WordFeatConfig, meta: FieldMeta):
        return cls(
            num_embeddings=meta.vocab_size,
            embedding_dim=config.embed_dim,
            embeds_weight=meta.pretrained_embeds_weight,
            init_range=config.embedding_init_range,
            unk_token_idx=meta.unk_token_idx,
            sparse=config.sparse,
        )

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embeds_weight: torch.Tensor,
        init_range: List[int],
        unk_token_idx: int,
        sparse: bool,
    ) -> None:
        EmbeddingBase.__init__(self, embedding_dim=embedding_dim)
        nn.Embedding.__init__(
            self, num_embeddings, embedding_dim, _weight=embeds_weight, sparse=sparse
        )
        if embeds_weight is None and init_range:
            self.weight.data.uniform_(init_range[0], init_range[1])
        # Initialize unk embedding with zeros
        # to guard the model against randomized decisions based on unknown words
        self.weight.data[unk_token_idx].fill_(0.0)
