#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.module_config import Activation, ModuleConfig
from pytext.models.module import create_module, Module
from pytext.optimizer import get_activation
from torch import Tensor

from .light_conv import LightweightConv


def mean(rep: Tensor, padding_mask: Optional[Tensor]):
    rep_sum = rep.sum(dim=1)  # B x T x C => B x C
    if padding_mask is not None:
        lengths = (~padding_mask).sum(dim=1).reshape(-1, 1)
    else:
        bsz, max_token_len, _embed_dim = rep.size()
        lengths = torch.full(
            (bsz, 1), max_token_len, dtype=torch.long, device=rep.device
        )

    return rep_sum / lengths


def pool(pooling_type: str, words: Tensor, encoder_padding_mask: Optional[Tensor]):
    # input dims: bsz * seq_len * num_filters
    if pooling_type == "mean":
        return mean(words, encoder_padding_mask)
    elif pooling_type == "max":
        return words.max(dim=1)[0]
    elif pooling_type == "none":
        return words
    else:
        raise NotImplementedError


class ConvLengthPredictionModule(Module):
    class Config(ModuleConfig):
        conv_dim: int = 128
        max_target_positions: int = 128
        length_dropout: float = 0.2
        kernel_sizes: List[int] = [3]
        glu: bool = True
        activation: Activation = Activation.GLU
        convolution_type: LightweightConv.Config = LightweightConv.Config()
        pooling_type: str = "mean"  # PoolingType.MEAN

    def __init__(
        self,
        embed_dim: int,
        conv_dim: int,
        max_target_positions: int,
        length_dropout: float,
        glu: bool,
        activation,
        pooling_type,
        conv_layers,
    ):
        super().__init__()
        self.length_dropout = length_dropout
        self.conv_layers = nn.ModuleList(conv_layers)
        self.glu = glu
        if glu:
            self.linear1 = nn.Linear(embed_dim, 2 * conv_dim)
        else:
            self.linear1 = nn.Linear(embed_dim, conv_dim)
        self.linear2 = nn.Linear(conv_dim, embed_dim)
        self.activation = get_activation(activation, dim=2)
        self.pooling_type = pooling_type
        self.lengths_pred = nn.Linear(embed_dim, max_target_positions)

    def forward(self, x: Tensor, encoder_padding_mask: Optional[Tensor] = None):
        for conv in self.conv_layers:
            residual = x
            x = self.linear1(x)
            x = self.activation(x)
            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
            # Input to conv() is T x B x C
            x = conv(x)
            x = self.linear2(x)
            x = F.dropout(x, p=self.length_dropout, training=self.training)
            x = residual + x
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = x.transpose(0, 1)  # T x B x C => B x T x C
        x = F.relu(x)
        lengths_enc = pool(self.pooling_type, x, encoder_padding_mask)
        predicted_lengths_logits = self.lengths_pred(lengths_enc)
        predicted_lengths = F.log_softmax(predicted_lengths_logits, dim=-1)
        return predicted_lengths, predicted_lengths_logits

    def create_eval_module(self):
        return self

    @classmethod
    def from_config(cls, config: Config, embed_dim: int):
        conv_layers = []
        for kernel_size in config.kernel_sizes:
            conv_layers.append(
                create_module(
                    config.convolution_type,
                    input_size=config.conv_dim,
                    kernel_size=kernel_size,
                    convolution_type="non-causal",
                )
            )

        return cls(
            embed_dim,
            config.conv_dim,
            config.max_target_positions,
            config.length_dropout,
            config.glu,
            config.activation,
            config.pooling_type,
            conv_layers,
        )


class MaskedLengthPredictionModule(Module):
    class Config(ModuleConfig):
        length_hidden_dim: int = 128
        max_target_positions: int = 128
        length_dropout: float = 0.2

    def __init__(
        self,
        embed_dim: int,
        length_hidden_dim: int,
        max_target_positions: int,
        length_dropout: float,
    ):
        super().__init__()
        self.lengths_linear = nn.Linear(embed_dim, length_hidden_dim)
        self.lengths_pred = nn.Linear(length_hidden_dim, max_target_positions)
        self.length_dropout = length_dropout

    def forward(
        self, x: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        avg_enc = mean(x.transpose(0, 1), encoder_padding_mask)
        lengths_enc = self.lengths_linear(avg_enc)
        lengths_enc = F.relu(lengths_enc)
        lengths_enc = F.dropout(
            lengths_enc, p=self.length_dropout, training=self.training
        )
        predicted_lengths_logits = self.lengths_pred(lengths_enc)
        # Softmax operations should be done in 32 bits when running
        # in FP16
        predicted_lengths = F.log_softmax(predicted_lengths_logits.float(), dim=-1).to(
            predicted_lengths_logits.dtype
        )
        return predicted_lengths, predicted_lengths_logits

    def create_eval_module(self):
        return self

    @classmethod
    def from_config(cls, config: Config, embed_dim: int):
        return cls(
            embed_dim,
            config.length_hidden_dim,
            config.max_target_positions,
            config.length_dropout,
        )
