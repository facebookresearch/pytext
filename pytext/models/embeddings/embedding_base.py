#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List

import torch.nn as nn
from pytext.models.module import Module


class EmbeddingBase(Module):
    """Base class for token level embedding modules.

    Args:
        embedding_dim (int): Size of embedding vector.

    Attributes:
        num_emb_modules (int): Number of ways to embed a token.
        embedding_dim (int): Size of embedding vector.

    """

    __EXPANSIBLE__ = True

    def __init__(self, embedding_dim: int):
        super().__init__()
        # By default has 1 embedding which is itself, for EmbeddingList, this num
        # can be greater than 1
        self.num_emb_modules = 1
        self.embedding_dim = embedding_dim

    def get_param_groups_for_optimizer(self) -> List[Dict[str, nn.Parameter]]:
        """
        Organize module parameters into param_groups (or layers), so the optimizer
        and / or schedulers can have custom behavior per layer.
        """
        return [dict(self.named_parameters())]
