#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn as nn

from .decoder_base import DecoderBase


class MLPDecoderQueryResponse(DecoderBase):
    """
    Implements a 'two-tower' MLP: one for query and one for response
    Used in search pairwise ranking: both pos_response and neg_response
    use the response-MLP
    """

    class Config(DecoderBase.Config):
        # Intermediate hidden dimensions
        hidden_dims: List[int] = []

    def __init__(self, config: Config, from_dim: int, to_dim: int) -> None:
        super().__init__(config)
        self.mlp_for_response = MLPDecoderQueryResponse.get_mlp(
            from_dim, to_dim, config.hidden_dims
        )
        self.mlp_for_query = MLPDecoderQueryResponse.get_mlp(
            from_dim, to_dim, config.hidden_dims
        )
        self.out_dim = (3, to_dim)

    @staticmethod
    def get_mlp(from_dim: int, to_dim: int, hidden_dims: List[int]):
        layers = []
        current_dim = from_dim
        for dim in hidden_dims or []:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            current_dim = dim
        layers.append(nn.Linear(current_dim, to_dim))
        return nn.Sequential(*layers)

    def forward(self, *x: List[torch.Tensor]) -> List[torch.Tensor]:
        output = []
        assert len(x) == 3
        output.append(self.mlp_for_response(x[0]))
        output.append(self.mlp_for_response(x[1]))
        output.append(self.mlp_for_query(x[2]))
        return output

    def get_decoder(self) -> List[nn.Module]:
        return [self.mlp_for_response, self.mlp_for_query]
