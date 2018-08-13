#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.onnx.operators


class SelfAttention(nn.Module):
    def __init__(self, n_input: int, n_attn: int, dropout_conf: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_conf)
        self.n_input = n_input
        self.n_attn = n_attn
        self.ws1 = nn.Linear(n_input, n_attn, bias=False)
        self.ws2 = nn.Linear(n_attn, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self, init_range: float = 0.1) -> None:
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
