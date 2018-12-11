#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.module_config import CNNParams
from pytext.utils import cuda_utils

from .representation_base import RepresentationBase


class BSeqCNNRepresentation(RepresentationBase):
    """
    This class is an implementation of the paper https://arxiv.org/pdf/1606.07783.
    It is a bidirectional CNN model that captures context like RNNs do.

    The module expects that input mini-batch is already padded.

    TODO: Current implementation has a single layer conv-maxpool operation.
    """

    class Config(RepresentationBase.Config):
        cnn = CNNParams()
        fwd_bwd_context_len: int = 5
        surrounding_context_len: int = 2

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        self.fwd_bwd_ctxt_len = config.fwd_bwd_context_len
        self.surr_ctxt_len = config.surrounding_context_len
        self.ctxt_pad_len = max(self.fwd_bwd_ctxt_len, self.surr_ctxt_len)
        self.padding_tensor = cuda_utils.Variable(
            torch.Tensor(1, self.fwd_bwd_ctxt_len, embed_dim), requires_grad=False
        )
        self.padding_tensor.fill_(0)

        bwd_convs, fwd_convs, surr_convs = [], [], []
        in_channels = 1
        out_channels = config.cnn.kernel_num
        kernel_sizes = config.cnn.kernel_sizes
        for k in kernel_sizes:
            bwd_convs.append(nn.Conv2d(in_channels, out_channels, (k, embed_dim)))
            fwd_convs.append(nn.Conv2d(in_channels, out_channels, (k, embed_dim)))
            surr_convs.append(nn.Conv2d(in_channels, out_channels, (k, embed_dim)))
        self.convs_bwd = nn.ModuleList(bwd_convs)
        self.convs_fwd = nn.ModuleList(fwd_convs)
        self.convs_surr = nn.ModuleList(surr_convs)

        # Token representation size with each context.
        token_rep_len = len(kernel_sizes) * out_channels
        self.bwd_fc = nn.Linear(token_rep_len, token_rep_len)
        self.fwd_fc = nn.Linear(token_rep_len, token_rep_len)
        self.surr_fc = nn.Linear(token_rep_len, token_rep_len)

        self.ctxt_pad = nn.ConstantPad1d((self.ctxt_pad_len, self.ctxt_pad_len), 0)

        self.representation_dim = 3 * len(kernel_sizes) * out_channels

    def forward(self, inputs: torch.Tensor, *args) -> torch.Tensor:
        inputs = self._preprocess_inputs(inputs)
        sent_rep = None
        for i in range(self.ctxt_pad_len, inputs.size()[1] - self.ctxt_pad_len):
            word_rep = torch.unsqueeze(self._word_forward(inputs, i), 1)
            if sent_rep is None:
                sent_rep = word_rep
            else:
                sent_rep = torch.cat((sent_rep, word_rep), dim=1)
        return sent_rep  # (N, W, 3*len(kernel_sizes)*out_channels)

    def _preprocess_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        # Transpose to make sentence dimension as last dimension of tensor,
        # so that we can add padding to the sentences. (N,W,D) -> (N,D,W)
        inputs = inputs.transpose(1, 2)

        # We need to pad when there isn't enough backward and forward context.
        # Pad sents_emb with max context length so that on-demand padding is not needed
        # to take care of boundary cases, e.g., backward context for the first word.
        inputs = self.ctxt_pad(inputs)

        # Transpose the tensor back to (N, W, D)
        return inputs.transpose(1, 2)

    def _word_forward(self, inputs: torch.Tensor, word_idx: int) -> torch.Tensor:
        # inputs -> (batch, words, embed_dim)
        start_idx = word_idx - self.fwd_bwd_ctxt_len
        word_with_bwd_context = inputs.narrow(1, start_idx, self.fwd_bwd_ctxt_len + 1)

        word_with_fwd_context = inputs.narrow(1, word_idx, self.fwd_bwd_ctxt_len + 1)

        start_idx = word_idx - self.surr_ctxt_len
        word_with_surr_context = inputs.narrow(1, start_idx, 2 * self.surr_ctxt_len + 1)

        padding = cuda_utils.Variable(
            torch.cat([self.padding_tensor] * inputs.size()[0])
        )
        conv_in_bwd_context = torch.cat((word_with_bwd_context, padding), dim=1)
        conv_in_fwd_context = torch.cat((padding, word_with_fwd_context), dim=1)

        bwd_ctxt_rep = self._conv_maxpool(conv_in_bwd_context, self.convs_bwd)
        fwd_ctxt_rep = self._conv_maxpool(conv_in_fwd_context, self.convs_fwd)
        surr_ctxt_rep = self._conv_maxpool(word_with_surr_context, self.convs_surr)

        # Full representation by combining all contextual represenations.
        return torch.cat(
            (
                self.bwd_fc(bwd_ctxt_rep),
                self.fwd_fc(fwd_ctxt_rep),
                self.surr_fc(surr_ctxt_rep),
            ),
            dim=1,
        )

    def _conv_maxpool(self, sents: torch.Tensor, convs: nn.ModuleList) -> torch.Tensor:
        # (N,Con,D) -> (N,Ci,Con,D); [Con = 2*context_len + 1], Ci = 1
        sents = sents.unsqueeze(1)

        # After conv operation, (N,Ci,Con,D) -> [(N,Co,W), ...]*len(Ks)
        conv_outs = [F.relu(conv(sents).squeeze(3)) for conv in convs]

        # [(N,Co,W), ...]*len(Ks)
        mp_outs = [
            F.max_pool1d(co, co.size(2), stride=1).squeeze(2) for co in conv_outs
        ]

        return torch.cat(mp_outs, 1)  # (N,len(Ks)*Co)
