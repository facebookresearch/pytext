#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.module_config import SlotAttentionType
from pytext.models.module import Module
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class SlotAttention(Module):
    class Config(ConfigBase):
        attn_dimension: int = 64
        attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION

    def __init__(self, config: Config, n_input: int, batch_first: bool = True) -> None:
        super().__init__()

        self.batch_first = batch_first
        self.attention_type = config.attention_type

        # attention can be in the form of h1'Wh2 ("multiply"),
        # g(h1;h2) ("add") or h1'h2 ("dot")
        if self.attention_type == SlotAttentionType.CONCAT:
            self.attention_add = nn.Sequential(
                nn.Linear(2 * n_input, config.attn_dimension, bias=False),
                nn.Tanh(),
                nn.Linear(config.attn_dimension, 1, bias=False),
            )
        elif self.attention_type == SlotAttentionType.MULTIPLY:
            self.attention_mult = nn.Linear(n_input, n_input, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(inputs, PackedSequence):
            inputs, lengths = pad_packed_sequence(inputs, batch_first=self.batch_first)
        # inputs -> bsz * num_words * dim
        size = inputs.size()

        # Tiling the full input on top of itself size[1] times
        exp_inputs_2 = inputs.unsqueeze(1).expand(size[0], size[1], size[1], size[2])
        if self.attention_type == SlotAttentionType.CONCAT:
            # Tiling each row on top of itself size[1] times
            exp_inputs_1 = inputs.unsqueeze(2).expand(
                size[0], size[1], size[1], size[2]
            )
            catted = torch.cat((exp_inputs_1, exp_inputs_2), 3)
            attn_weights_add = F.softmax(
                self.attention_add(catted).squeeze(3), dim=2
            ).unsqueeze(2)
            context_add = torch.matmul(attn_weights_add, exp_inputs_2).squeeze(2)
            output = torch.cat((inputs, context_add), 2)
        elif (
            self.attention_type == SlotAttentionType.MULTIPLY
            or self.attention_type == SlotAttentionType.DOT
        ):
            attended = (
                inputs
                if self.attention_type == SlotAttentionType.DOT
                else self.attention_mult(inputs)
            )
            attn_weights_mult = F.softmax(
                torch.matmul(inputs, torch.transpose(attended, 1, 2)), dim=2
            ).unsqueeze(2)
            context_mult = torch.matmul(attn_weights_mult, exp_inputs_2).squeeze(2)
            output = torch.cat((inputs, context_mult), 2)
        else:
            output = inputs

        return output
