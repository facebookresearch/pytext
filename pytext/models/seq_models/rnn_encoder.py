#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.utils.usage import log_class_usage
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import PyTextSeq2SeqModule


class BiLSTM(torch.nn.Module):
    """Wrapper for nn.LSTM

    Differences include:
    * weight initialization
    * the bidirectional option makes the first layer bidirectional only
    (and in that case the hidden dim is divided by 2)
    """

    @staticmethod
    def LSTM(input_size, hidden_size, **kwargs):
        m = torch.nn.LSTM(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)
        return m

    def __init__(self, num_layers, bidirectional, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            assert hidden_dim % 2 == 0, "hidden_dim should be even if bidirectional"
        self.hidden_dim = hidden_dim

        self.layers = torch.nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = bidirectional and layer == 0
            if is_layer_bidirectional:
                assert hidden_dim % 2 == 0, (
                    "hidden_dim must be even if bidirectional "
                    "(to be divided evenly between directions)"
                )
            self.layers.append(
                BiLSTM.LSTM(
                    embed_dim if layer == 0 else hidden_dim,
                    hidden_dim // 2 if is_layer_bidirectional else hidden_dim,
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=is_layer_bidirectional,
                )
            )
        log_class_usage(__class__)

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor,
        enforce_sorted: bool = True,
    ):
        # enforce_sorted is set to True by default to force input lengths
        # are sorted in a descending order when pack padded sequence.
        bsz = embeddings.size()[1]

        # Generate packed seq to deal with varying source seq length
        # packed_input is of type PackedSequence, which consists of:
        # element [0]: a tensor, the packed data, and
        # element [1]: a list of integers, the batch size for each step
        packed_input = pack_padded_sequence(
            embeddings, lengths, enforce_sorted=enforce_sorted
        )

        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = embeddings.new_full((2, bsz, self.hidden_dim // 2), 0)
                c0 = embeddings.new_full((2, bsz, self.hidden_dim // 2), 0)
            else:
                h0 = embeddings.new_full((1, bsz, self.hidden_dim), 0)
                c0 = embeddings.new_full((1, bsz, self.hidden_dim), 0)

            # apply LSTM along entire sequence
            current_output, (h_last, c_last) = rnn_layer(packed_input, (h0, c0))

            # final state shapes: (bsz, hidden_dim)
            if self.bidirectional and i == 0:
                # concatenate last states for forward and backward LSTM
                h_last = torch.cat((h_last[0, :, :], h_last[1, :, :]), dim=1)
                c_last = torch.cat((c_last[0, :, :], c_last[1, :, :]), dim=1)
            else:
                h_last = h_last.squeeze(dim=0)
                c_last = c_last.squeeze(dim=0)

            final_hiddens.append(h_last)
            final_cells.append(c_last)

            packed_input = current_output

        # Reshape to [num_layer, batch_size, hidden_dim]
        final_hidden_size_list: List[int] = final_hiddens[0].size()
        final_hidden_size: Tuple[int, int] = (
            final_hidden_size_list[0],
            final_hidden_size_list[1],
        )
        final_hiddens = torch.cat(final_hiddens, dim=0).view(
            self.num_layers, *final_hidden_size
        )
        final_cell_size_list: List[int] = final_cells[0].size()
        final_cell_size: Tuple[int, int] = (
            final_cell_size_list[0],
            final_cell_size_list[1],
        )
        final_cells = torch.cat(final_cells, dim=0).view(
            self.num_layers, *final_cell_size
        )

        #  [max_seqlen, batch_size, hidden_dim]
        unpacked_output, _ = pad_packed_sequence(packed_input)

        return (unpacked_output, final_hiddens, final_cells)


class LSTMSequenceEncoder(PyTextSeq2SeqModule):
    """RNN encoder using nn.LSTM for cuDNN support / ONNX exportability."""

    class Config(ConfigBase):
        embed_dim: int = 512
        hidden_dim: int = 512
        num_layers: int = 1
        dropout_in: float = 0.1
        dropout_out: float = 0.1
        bidirectional: bool = False

    def __init__(
        self, embed_dim, hidden_dim, num_layers, dropout_in, dropout_out, bidirectional
    ):

        super().__init__()
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers: int = num_layers

        self.word_dim = embed_dim

        self.bilstm = BiLSTM(
            num_layers=num_layers,
            bidirectional=bidirectional,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_out,
        )
        log_class_usage(__class__)

    @classmethod
    def from_config(cls, config):
        return cls(**config._asdict())

    def forward(
        self, src_tokens: torch.Tensor, embeddings: torch.Tensor, src_lengths
    ) -> Dict[str, torch.Tensor]:

        x = F.dropout(embeddings, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        unpacked_output, final_hiddens, final_cells = self.bilstm(
            embeddings=x, lengths=src_lengths
        )

        return {
            "unpacked_output": unpacked_output,
            "final_hiddens": final_hiddens,
            "final_cells": final_cells,
            "src_lengths": src_lengths,
            "src_tokens": src_tokens,
            "embeddings": embeddings,
        }

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def tile_encoder_out(
        self, beam_size: int, encoder_out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        tiled_encoder_out = encoder_out["unpacked_output"].expand(-1, beam_size, -1)
        hiddens = encoder_out["final_hiddens"]
        tiled_hiddens: List[torch.Tensor] = []
        for i in range(self.num_layers):
            tiled_hiddens.append(hiddens[i].expand(beam_size, -1))
        cells = encoder_out["final_cells"]
        tiled_cells: List[torch.Tensor] = []
        for i in range(self.num_layers):
            tiled_cells.append(cells[i].expand(beam_size, -1))
        # tiled_src_lengths = encoder_out["src_lengths"].expand(-1, beam_size, -1)
        return {
            "unpacked_output": tiled_encoder_out,
            "final_hiddens": torch.stack(tiled_hiddens, dim=0),
            "final_cells": torch.stack(tiled_cells, dim=0),
            "src_lengths": encoder_out["src_lengths"],
            "src_tokens": encoder_out["src_tokens"],
        }
