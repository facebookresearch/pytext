#!/usr/bin/env python3

from typing import Tuple

import torch
from pytext.config.field_config import FeatureConfig

from .token_embedding import TokenEmbedding


class SharedTokenEmbedding(TokenEmbedding):
    """Shared token embeddings.

    Used to embed a tuple of inputs in the same way.
    """

    class Config(FeatureConfig):
        pass

    def forward(
        self, tokens: Tuple[torch.Tensor, ...], lengths: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(
            super(SharedTokenEmbedding, self).forward(toks, lens)
            for toks, lens in zip(tokens, lengths)
        )
