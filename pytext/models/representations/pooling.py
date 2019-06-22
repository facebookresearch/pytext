#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.onnx.operators
from pytext.config import ConfigBase
from pytext.models.module import Module


class SelfAttention(Module):
    class Config(ConfigBase):
        attn_dimension: int = 64
        dropout: float = 0.4

    def __init__(self, config: Config, n_input: int) -> None:
        super().__init__(config)

        self.dropout = nn.Dropout(config.dropout)
        self.n_input = n_input
        self.n_attn = config.attn_dimension
        self.ws1 = nn.Linear(n_input, self.n_attn, bias=False)
        self.ws2 = nn.Linear(self.n_attn, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self, init_range: float = 0.1) -> None:
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(
        self, inputs: torch.Tensor, seq_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        # size: (bsz, sent_len, rep_dim)
        size = torch.onnx.operators.shape_as_tensor(inputs)

        flat_2d_shape = torch.cat((torch.LongTensor([-1]), size[2].view(1)))
        compressed_emb = torch.onnx.operators.reshape_from_tensor_shape(
            inputs, flat_2d_shape
        )  # (bsz * sent_len, rep_len)
        hbar = self.tanh(
            self.ws1(self.dropout(compressed_emb))
        )  # (bsz * sent_len, attention_dim)
        alphas = self.ws2(hbar)  # (bsz * sent_len, 1)
        alphas = torch.onnx.operators.reshape_from_tensor_shape(
            alphas, size[:2]
        )  # (bsz, sent_len)
        alphas = self.softmax(alphas)  # (bsz, sent_len)

        # (bsz, rep_dim)
        return torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)


class MaxPool(Module):
    def __init__(self, config: Module.Config, n_input: int) -> None:
        super().__init__(config)

    def forward(
        self, inputs: torch.Tensor, seq_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        return torch.max(inputs, 1)[0]


class MeanPool(Module):
    def __init__(self, config: Module.Config, n_input: int) -> None:
        super().__init__(config)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        return torch.sum(inputs, 1) / seq_lengths.unsqueeze(1).float()


class NoPool(Module):
    def __init__(self, config: Module.Config, n_input: int) -> None:
        super().__init__(config)

    def forward(
        self, inputs: torch.Tensor, seq_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        return inputs


class BoundaryPool(Module):
    class Config(ConfigBase):
        # first, last, firstlast
        boundary_type: str = "first"

    def __init__(self, config: Config, n_input: int) -> None:
        super().__init__(config)
        self.boundary_type = config.boundary_type

    def forward(
        self, inputs: torch.Tensor, seq_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        max_len = inputs.size()[1]
        if self.boundary_type == "first":
            return inputs[:, 0, :]
        elif self.boundary_type == "last":
            # could only have the bos values if add_bos or add_eos as False
            # should not reach here if the eos is not added.
            assert max_len > 1
            return inputs[:, max_len - 1, :]
        elif self.boundary_type == "firstlast":
            assert max_len > 1
            # merge from embed_dim into 2*emded_dim
            return torch.cat((inputs[:, 0, :], inputs[:, max_len - 1, :]), dim=1)
        else:
            raise Exception("Unknown configuration type {}".format(self.boundary_type))


class LastTimestepPool(Module):
    def __init__(self, config: Module.Config, n_input: int) -> None:
        super().__init__(config)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        # inputs: (bsz, max_len, dim)
        # seq_lengths: (bsz,)

        if torch._C._get_tracing_state():
            # if it is exporting, the batch size = 1, so we return the last hidden state
            # by returning the last dimension to avoid introducing extra operators
            assert inputs.shape[0] == 1
            return inputs[:, -1, :]
        bsz, _, dim = inputs.shape
        idx = seq_lengths.unsqueeze(1).expand(bsz, dim).unsqueeze(1)
        return inputs.gather(1, idx - 1).squeeze(1)
