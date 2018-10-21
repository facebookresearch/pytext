#!/usr/bin/env python3

from typing import Tuple

import torch

from .token_embedding import TokenEmbedding


class SharedTokenEmbedding(TokenEmbedding):
    """Shared token embeddings.

    Used to embed a tuple of inputs in the same way.
    """

    def forward(
        self, tokens: Tuple[torch.Tensor, ...], lengths: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(
            super(SharedTokenEmbedding, self).forward(toks, lens)
            for toks, lens in zip(tokens, lengths)
        )
