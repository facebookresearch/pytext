#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.module_config import CNNParams
from pytext.models.representations.representation_base import RepresentationBase


class ContextualWordConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int]):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, k, padding=k - 1)
                for k in kernel_sizes
            ]
        )
        token_rep_size = len(kernel_sizes) * out_channels
        self.fc = nn.Linear(token_rep_size, token_rep_size)

    def forward(self, words: torch.Tensor):
        words = words.transpose(1, 2)
        conv_outs = [F.relu(conv(words)) for conv in self.convs]
        mp_outs = [self.max_pool(co).squeeze(2) for co in conv_outs]
        return self.fc(torch.cat(mp_outs, 1))


class BSeqCNNRepresentation(RepresentationBase):
    """
    This class is an implementation of the paper https://arxiv.org/pdf/1606.07783.
    It is a bidirectional CNN model that captures context like RNNs do.

    The module expects that input mini-batch is already padded.

    TODO: Current implementation has a single layer conv-maxpool operation.
    """

    class Config(RepresentationBase.Config):
        cnn: CNNParams = CNNParams()
        fwd_bwd_context_len: int = 5
        surrounding_context_len: int = 2

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        self.fwd_bwd_ctxt_len = config.fwd_bwd_context_len
        self.surr_ctxt_len = config.surrounding_context_len
        self.ctxt_pad_len = max(self.fwd_bwd_ctxt_len, self.surr_ctxt_len)

        out_channels = config.cnn.kernel_num
        kernel_sizes = config.cnn.kernel_sizes
        self.backward_conv = ContextualWordConvolution(
            embed_dim, out_channels, kernel_sizes
        )
        self.forward_conv = ContextualWordConvolution(
            embed_dim, out_channels, kernel_sizes
        )
        self.surround_conv = ContextualWordConvolution(
            embed_dim, out_channels, kernel_sizes
        )

        self.ctxt_pad = nn.ConstantPad1d((self.ctxt_pad_len, self.ctxt_pad_len), 0)

        self.representation_dim = 3 * len(kernel_sizes) * out_channels

    def forward(self, inputs: torch.Tensor, *args) -> torch.Tensor:
        inputs = self._preprocess_inputs(inputs)
        word_reps = []
        for i in range(self.ctxt_pad_len, inputs.size()[1] - self.ctxt_pad_len):
            word_rep = self._word_forward(inputs, i).unsqueeze(1)
            word_reps.append(word_rep)

        sent_rep = torch.cat(word_reps, dim=1)

        return sent_rep  # (N, W, 3*len(kernel_sizes)*out_channels)

    def _preprocess_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        # Transpose to make sentence dimension as last dimension of tensor,
        # so that we can add padding to the sentences. (N, W, D) -> (N, D, W)
        inputs = inputs.transpose(1, 2)

        # We need to pad when there isn't enough backward and forward context.
        # Pad sents_emb with max context length so that on-demand padding is not needed
        # to take care of boundary cases, e.g., backward context for the first word.
        inputs = self.ctxt_pad(inputs.unsqueeze(1)).squeeze(1)

        # Transpose the tensor back to (N, W, D)
        return inputs.transpose(1, 2)

    def _word_forward(self, inputs: torch.Tensor, word_idx: int) -> torch.Tensor:
        # inputs -> (batch, words, embed_dim)
        start_idx = word_idx - self.fwd_bwd_ctxt_len
        word_with_bwd_context = inputs.narrow(1, start_idx, self.fwd_bwd_ctxt_len)

        word_with_fwd_context = inputs.narrow(1, word_idx, self.fwd_bwd_ctxt_len)

        start_idx = word_idx - self.surr_ctxt_len
        word_with_surr_context = inputs.narrow(1, start_idx, 2 * self.surr_ctxt_len)

        return torch.cat(
            (
                self.backward_conv(word_with_bwd_context),
                self.forward_conv(word_with_fwd_context),
                self.surround_conv(word_with_surr_context),
            ),
            dim=1,
        )
