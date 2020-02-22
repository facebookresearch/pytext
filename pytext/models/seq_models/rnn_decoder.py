#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils as fairseq_utils
from pytext.config import ConfigBase
from pytext.models.seq_models.base import PyTextSeq2SeqModule
from pytext.utils.usage import log_class_usage
from torch import nn

from .attention import DotAttention
from .base import PlaceholderIdentity, PyTextIncrementalDecoderComponent


class DecoderWithLinearOutputProjection(PyTextSeq2SeqModule):
    """
    Common super class for decoder networks with output projection layers.
    """

    def __init__(self, out_vocab_size, out_embed_dim=512):
        super().__init__()
        self.linear_projection = nn.Linear(out_embed_dim, out_vocab_size)
        self.reset_parameters()
        log_class_usage(__class__)

    def reset_parameters(self):
        nn.init.uniform_(self.linear_projection.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear_projection.bias)

    def forward(
        self,
        input_tokens,
        encoder_out: Dict[str, torch.Tensor],
        incremental_state: Optional[Dict[str, torch.Tensor]] = None,
        timestep: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, features = self.forward_unprojected(
            input_tokens, encoder_out, incremental_state
        )
        logits = self.linear_projection(x)
        return logits, features

    def forward_unprojected(self, input_tokens, encoder_out, incremental_state=None):
        """Forward pass through the decoder without output projection."""
        raise NotImplementedError()


class RNNDecoderBase(PyTextIncrementalDecoderComponent):
    """
    RNN decoder with multihead attention. Attention is calculated using encoder
    output and output of decoder's first RNN layerself. Attention is applied
    after first RNN layer and concatenated to input of subsequent layers.
    """

    class Config(ConfigBase):
        encoder_hidden_dim: int = 512
        embed_dim: int = 512
        hidden_dim: int = 512
        out_embed_dim: int = 512
        cell_type: str = "lstm"
        num_layers: int = 1
        dropout_in: float = 0.1
        dropout_out: float = 0.1
        attention_type: str = "dot"
        attention_heads: int = 8
        first_layer_attention: bool = False
        averaging_encoder: bool = False

    @classmethod
    def from_config(cls, config, out_vocab_size, target_embedding):
        return cls(out_vocab_size, target_embedding, **config._asdict())

    def __init__(
        self,
        embed_tokens,
        encoder_hidden_dim,
        embed_dim,
        hidden_dim,
        out_embed_dim,
        cell_type,
        num_layers,
        dropout_in,
        dropout_out,
        attention_type,
        attention_heads,
        first_layer_attention,
        averaging_encoder,
    ):
        encoder_hidden_dim = max(1, encoder_hidden_dim)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_embed_dim = out_embed_dim
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.attention_type = attention_type
        self.attention_heads = attention_heads
        self.first_layer_attention = first_layer_attention
        self.embed_tokens = embed_tokens

        self.hidden_dim = hidden_dim
        self.averaging_encoder = averaging_encoder

        if cell_type == "lstm":
            cell_class = torch.nn.LSTMCell
        else:
            raise RuntimeError("Cell type not supported")

        self.change_hidden_dim = hidden_dim != encoder_hidden_dim
        if self.change_hidden_dim:
            hidden_init_fc_list = []
            cell_init_fc_list = []
            for _ in range(num_layers):
                hidden_init_fc_list.append(nn.Linear(encoder_hidden_dim, hidden_dim))
                cell_init_fc_list.append(nn.Linear(encoder_hidden_dim, hidden_dim))
            self.hidden_init_fc_list = nn.ModuleList(hidden_init_fc_list)
            self.cell_init_fc_list = nn.ModuleList(cell_init_fc_list)
        else:
            # Empty module lists to appease Torchscript
            self.hidden_init_fc_list = nn.ModuleList([])
            self.cell_init_fc_list = nn.ModuleList([])

        if attention_type == "dot":
            self.attention = DotAttention(
                decoder_hidden_state_dim=hidden_dim, context_dim=encoder_hidden_dim
            )
        else:
            raise RuntimeError(f"Attention type {attention_type} not supported")

        self.combined_output_and_context_dim = self.attention.context_dim + hidden_dim

        layers = []
        for layer in range(num_layers):
            if layer == 0:
                cell_input_dim = embed_dim
            else:
                cell_input_dim = hidden_dim

            # attention applied to first layer always.
            if self.first_layer_attention or layer == 0:
                cell_input_dim += self.attention.context_dim
            layers.append(cell_class(input_size=cell_input_dim, hidden_size=hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)

        if self.combined_output_and_context_dim != out_embed_dim:
            self.additional_fc = nn.Linear(
                self.combined_output_and_context_dim, out_embed_dim
            )
        else:
            # Using identity layer in place of the bottleneck simplifies torchscript
            # compatibility.
            self.additional_fc = PlaceholderIdentity()
        log_class_usage(__class__)

    def forward_unprojected(
        self,
        input_tokens,
        encoder_out: Dict[str, torch.Tensor],
        incremental_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if incremental_state is not None and len(incremental_state) > 0:
            input_tokens = input_tokens[:, -1:]
        bsz, seqlen = input_tokens.size()

        # get outputs from encoder
        encoder_outs = encoder_out["unpacked_output"]
        src_lengths = encoder_out["src_lengths"]

        # embed tokens
        x = self.embed_tokens([[input_tokens]])
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = self._get_cached_state(incremental_state)
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            # first time step, initialize previous states
            if incremental_state is None:
                incremental_state = {}
            self._init_prev_states(encoder_out, incremental_state)
            init_state = self._get_cached_state(incremental_state)
            assert init_state is not None
            prev_hiddens, prev_cells, input_feed = init_state

        outs = []
        attn_scores_per_step: List[torch.Tensor] = []
        next_hiddens: List[torch.Tensor] = []
        next_cells: List[torch.Tensor] = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            step_input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(step_input, (prev_hiddens[i], prev_cells[i]))

                if self.first_layer_attention and i == 0:
                    # tgt_len is 1 in decoder and squeezed for both matrices
                    # input_feed.shape = tgt_len X bsz X embed_dim
                    # step_attn_scores.shape = src_len X tgt_len X bsz
                    input_feed, step_attn_scores = self.attention(
                        hidden, encoder_outs, src_lengths
                    )

                # hidden state becomes the input to the next layer
                layer_output = F.dropout(
                    hidden, p=self.dropout_out, training=self.training
                )

                step_input = layer_output

                if self.first_layer_attention:
                    step_input = torch.cat((step_input, input_feed), dim=1)

                # save state for next time step
                next_hiddens.append(hidden)
                next_cells.append(cell)

            if not self.first_layer_attention:
                input_feed, step_attn_scores = self.attention(
                    hidden, encoder_outs, src_lengths
                )

                attn_scores_per_step.append(step_attn_scores)
            combined_output_and_context = torch.cat((hidden, input_feed), dim=1)

            # save final output
            outs.append(combined_output_and_context)

            # update hidden states for next timestep
            prev_hiddens = torch.stack(next_hiddens, 0)
            prev_cells = torch.stack(next_cells, 0)
            next_hiddens = []
            next_cells = []

        attn_scores = torch.stack(attn_scores_per_step, dim=1)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # cache previous states
        self._set_cached_state(
            incremental_state, (prev_hiddens, prev_cells, input_feed)
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(
            seqlen, bsz, self.combined_output_and_context_dim
        )

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # bottleneck layer
        x = self.additional_fc(x)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        return (
            x,
            {
                "attn_scores": attn_scores,
                "src_tokens": encoder_out["src_tokens"],
                "src_lengths": encoder_out["src_lengths"],
            },
        )

    def reorder_incremental_state(
        self, incremental_state: Dict[str, torch.Tensor], new_order
    ):
        """Reorder buffered internal state (for incremental generation)."""
        assert incremental_state is not None
        hiddens = self.get_incremental_state(incremental_state, "cached_hiddens")
        assert hiddens is not None
        cells = self.get_incremental_state(incremental_state, "cached_cells")
        assert cells is not None
        feeds = self.get_incremental_state(incremental_state, "cached_feeds")
        assert feeds is not None

        self.set_incremental_state(
            incremental_state, "cached_hiddens", hiddens.index_select(1, new_order)
        )
        self.set_incremental_state(
            incremental_state, "cached_cells", cells.index_select(1, new_order)
        )
        self.set_incremental_state(
            incremental_state, "cached_feeds", feeds.index_select(0, new_order)
        )

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def _init_prev_states(
        self,
        encoder_out: Dict[str, torch.Tensor],
        incremental_state: Dict[str, torch.Tensor],
    ) -> None:
        encoder_output = encoder_out["unpacked_output"]
        final_hiddens = encoder_out["final_hiddens"]
        prev_cells = encoder_out["final_cells"]
        if self.averaging_encoder:
            # Use mean encoder hidden states
            prev_hiddens = torch.stack(
                [torch.mean(encoder_output, 0)] * self.num_layers, dim=0
            )
        else:
            # Simply return the final state of each layer
            prev_hiddens = final_hiddens

        if self.change_hidden_dim:
            transformed_hiddens: List[torch.Tensor] = []
            transformed_cells: List[torch.Tensor] = []
            i: int = 0
            for hidden_init_fc, cell_init_fc in zip(
                self.hidden_init_fc_list, self.cell_init_fc_list
            ):
                transformed_hiddens.append(hidden_init_fc(prev_hiddens[i]))
                transformed_cells.append(cell_init_fc(prev_cells[i]))
                i += 1
            use_hiddens = torch.stack(transformed_hiddens, dim=0)
            use_cells = torch.stack(transformed_cells, dim=0)
        else:
            use_hiddens = prev_hiddens
            use_cells = prev_cells

        assert self.attention.context_dim
        initial_attn_context = torch.zeros(
            self.attention.context_dim, device=encoder_output.device
        )
        batch_size = encoder_output.size(1)

        self.set_incremental_state(incremental_state, "cached_hiddens", use_hiddens)
        self.set_incremental_state(incremental_state, "cached_cells", use_cells)
        self.set_incremental_state(
            incremental_state,
            "cached_feeds",
            initial_attn_context.expand(batch_size, self.attention.context_dim),
        )

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return fairseq_utils.log_softmax(logits, dim=-1)
        else:
            return fairseq_utils.softmax(logits, dim=-1)

    def _get_cached_state(self, incremental_state: Optional[Dict[str, torch.Tensor]]):
        if incremental_state is None or len(incremental_state) == 0:
            return None
        hiddens = self.get_incremental_state(incremental_state, "cached_hiddens")
        assert hiddens is not None
        cells = self.get_incremental_state(incremental_state, "cached_cells")
        assert cells is not None
        feeds = self.get_incremental_state(incremental_state, "cached_feeds")
        assert feeds is not None
        return (hiddens, cells, feeds)

    def _set_cached_state(
        self,
        incremental_state: Optional[Dict[str, torch.Tensor]],
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        if incremental_state is None:
            return
        (hiddens, cells, feeds) = state
        self.set_incremental_state(incremental_state, "cached_hiddens", hiddens)
        self.set_incremental_state(incremental_state, "cached_cells", cells)
        self.set_incremental_state(incremental_state, "cached_feeds", feeds)


class RNNDecoder(RNNDecoderBase, DecoderWithLinearOutputProjection):
    def __init__(
        self,
        out_vocab_size,
        embed_tokens,
        encoder_hidden_dim,
        embed_dim,
        hidden_dim,
        out_embed_dim,
        cell_type,
        num_layers,
        dropout_in,
        dropout_out,
        attention_type,
        attention_heads,
        first_layer_attention,
        averaging_encoder,
    ):
        DecoderWithLinearOutputProjection.__init__(
            self, out_vocab_size, out_embed_dim=out_embed_dim
        )
        RNNDecoderBase.__init__(
            self,
            embed_tokens,
            encoder_hidden_dim,
            embed_dim,
            hidden_dim,
            out_embed_dim,
            cell_type,
            num_layers,
            dropout_in,
            dropout_out,
            attention_type,
            attention_heads,
            first_layer_attention,
            averaging_encoder,
        )
        log_class_usage(__class__)
