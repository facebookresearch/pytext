#!/usr/bin/env python3

from typing import Tuple

import torch
import torch.nn as nn
from pytext.models.configs.embedding_config import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)

from .char_embedding import CharacterEmbedding
from .dict_embedding import DictEmbedding


class TokenEmbedding(nn.Module):
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

    def __init__(
        self,
        word_emb_config: WordEmbeddingConfig,
        dict_emb_config: DictEmbeddingConfig,
        char_emb_config: CharacterEmbeddingConfig,
    ) -> None:
        super().__init__()
        self.embedding_dim = 0

        if word_emb_config:
            self.word_embed = nn.Embedding(
                word_emb_config.vocab_size,
                word_emb_config.embedding_dim,
                _weight=word_emb_config.pretrained_embeds_weight,
                sparse=word_emb_config.sparse_grad,
            )
            # Initialize unk embedding with zeros
            # to guard the model against randomized decisions based on unknown words
            self.word_embed.weight.data[word_emb_config.unk_idx].fill_(0.0)
            self.embedding_dim += word_emb_config.embedding_dim
            self.token_vocab_size = word_emb_config.vocab_size
            self.word_embed.weight.requires_grad = not word_emb_config.freeze_embeds

        if dict_emb_config:
            self.dict_embed = DictEmbedding(dict_emb_config)
            self.embedding_dim += dict_emb_config.embedding_dim
            self.dict_vocab_size = dict_emb_config.vocab_size

        if char_emb_config:
            self.char_embed = CharacterEmbedding(char_emb_config)
            self.embedding_dim += char_emb_config.out_channels * len(
                char_emb_config.kernel_sizes
            )

        if self.embedding_dim == 0:
            raise ValueError("At least one embedding config must be provided.")

    def forward(
        self,
        tokens: torch.Tensor,
        dict_feat: Tuple[torch.Tensor, ...] = None,
        cap_feat: Tuple[torch.Tensor, ...] = None,
        chars: torch.Tensor = None,
    ) -> torch.Tensor:
        # tokens dim: (bsz, max_seq_len) -> (bsz, max_seq_len, dim)
        embeddings = []

        if hasattr(self, "word_embed"):
            word_emb = self.word_embed(tokens)
            embeddings.append(word_emb)

        if hasattr(self, "dict_embed"):
            if not dict_feat:
                raise ValueError("dict_feat argument is None.")
            dict_ids, dict_weights, dict_lengths = dict_feat
            dict_emb = self.dict_embed(tokens, dict_ids, dict_weights, dict_lengths)
            embeddings.append(dict_emb)

        if hasattr(self, "char_embed"):
            if chars is None:
                raise ValueError("char_feat argument is None")
            char_emb = self.char_embed(chars)
            embeddings.append(char_emb)

        return torch.cat(embeddings, 2)
