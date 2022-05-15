#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import nn, Tensor


@with_incremental_state
class LunarMultiheadAttention(nn.Module):
    """Lunar Multi-headed attention.
    See "Linformer: Self-Attention with Linear Complexity" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_pheads,
        dropout=0.0,
        bias=True,
        self_attention=False,
        encoder_decoder_attention=False,
        tie_kv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_pheads = num_pheads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        self.phead_dim = embed_dim // num_pheads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert (
            self.phead_dim * num_pheads == self.embed_dim
        ), "projected embed_dim must be divisible by num_pheads"
        self.scaling = self.head_dim**-0.5
        self.pscaling = self.phead_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.pq_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if tie_kv:
            self.pc_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.pk_proj = self.k_proj = None
            self.pv_proj = self.v_proj = None
        else:
            self.pk_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.pv_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.pc_proj = self.c_proj = None

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
        raise NotImplementedError("onnx for linear attention not implemented")

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True
        raise NotImplementedError("TPU for linear attention not implemented")

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.pq_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        if self.pq_proj.bias is not None:
            nn.init.constant_(self.pq_proj.bias, 0.0)
            nn.init.constant_(self.q_proj.bias, 0.0)

        if self.pc_proj is not None:
            nn.init.xavier_uniform_(self.pc_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.c_proj.weight, gain=gain)
            if self.pc_proj.bias is not None:
                nn.init.constant_(self.pc_proj.bias, 0.0)
                nn.init.constant_(self.c_proj.bias, 0.0)
        else:
            nn.init.xavier_uniform_(self.pk_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.pv_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
            if self.pk_proj.bias is not None:
                nn.init.constant_(self.pk_proj.bias, 0.0)
                nn.init.constant_(self.pv_proj.bias, 0.0)
                nn.init.constant_(self.k_proj.bias, 0.0)
                nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _compute_pcontext(self, pquery, context, context_padding_mask):
        # N x B x D
        len, bsz, dim = context.size()
        if self.pc_proj is not None:
            c = self.pc_proj(context)
            # N x B x D -> N x B*H x K
            k = v = c.view(len, bsz * self.num_pheads, self.phead_dim)
        else:
            # N x B x D -> N x B*H x K
            k = self.pk_proj(context).view(len, bsz * self.num_pheads, self.phead_dim)
            v = self.pv_proj(context).view(len, bsz * self.num_pheads, self.phead_dim)

        # N x B*H x K -> B*H x K x N
        k = k.permute(1, 2, 0)
        # N x B*H x K -> B*H x N x K
        v = v.transpose(0, 1)

        plen = pquery.size(0)
        # L x B x D -> L x B*H x K
        pq = self.pq_proj(pquery).view(plen, bsz * self.num_pheads, self.phead_dim)
        # L x B*H x K -> B*H x L x K
        pq = pq.transpose(0, 1) * self.pscaling
        # B*H x L x N
        pqk = torch.bmm(pq, k)
        if context_padding_mask is not None:
            pqk = pqk.view(bsz, self.num_pheads, plen, len)
            pqk = pqk.masked_fill(
                context_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            pqk = pqk.view(bsz * self.num_pheads, plen, len)

        pqk = F.softmax(pqk, dim=-1)
        pqk = self.dropout_module(pqk)
        # B*H x L x K
        pc = torch.bmm(pqk, v)
        # B*H x L x K -> L x B*H x K -> L x B x D
        pc = pc.transpose(0, 1).contiguous().view(plen, bsz, dim)
        return pc

    def compute_pcontext(
        self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        static_context: bool = False,
    ) -> Union[Tensor, None]:

        if context is None:
            return context
        else:
            return self._compute_pcontext(pquery, context, context_padding_mask)

    def forward(
        self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        pcontext_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_context: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            context_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert (
            not self.self_attention or incremental_state is None
        ), "For incremental self attention (causal attention), please use LunarCausalAttention"

        if self.self_attention:
            context = query

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_context:
                    assert self.encoder_decoder_attention and not self.self_attention
                    context = None
        else:
            saved_state = None

        # L x B x D
        pcontext = self.compute_pcontext(
            query,
            pquery,
            context,
            context_padding_mask,
            incremental_state,
            static_context,
        )

        key_padding_mask = pcontext_padding_mask

        q = self.q_proj(query)
        if pcontext is None:
            assert context is None
            k = v = None
        elif self.c_proj is not None:
            k = v = (
                self.c_proj(pcontext)
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            k = (
                self.k_proj(pcontext)
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            v = (
                self.v_proj(pcontext)
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        q = q * self.scaling
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            # pcontext are stored with shape (bsz, proj_len, model_dim)
            if "prev_pcontext" in saved_state:
                # TODO save prev_pcontext for causal attention
                _prev_pcontext = saved_state["prev_pcontext"]
                assert _prev_pcontext is not None
                prev_pcontext = _prev_pcontext.transpose(0, 1)
                if static_context:
                    pcontext = prev_pcontext
                else:
                    raise RuntimeError("pcontext error")

            assert k is not None and v is not None and pcontext is not None
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_pcontext"] = pcontext.transpose(0, 1)
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None and v is not None and pcontext is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = LunarMultiheadAttention.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz
        )

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, pcontext, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


@with_incremental_state
class LunarCausalAttention(nn.Module):
    """Lunar Causal attention.
    See "Linformer: Self-Attention with Linear Complexity" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        tie_kv=True,
        parallel=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.pq_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.pc_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if tie_kv:
            self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = self.v_proj = None
        else:
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.c_proj = None

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
        raise NotImplementedError("onnx for linear attention not implemented")

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True
        raise NotImplementedError("TPU for linear attention not implemented")

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.pq_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.pc_proj.weight, gain=gain)
        if self.c_proj is not None:
            nn.init.xavier_uniform_(self.c_proj.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _compute_pattention(self, pq, context, key_padding_mask):
        # N x B x D
        len, bsz, dim = context.size()
        # N x B x D
        k = self.pc_proj(context)
        # N x B*H x K -> B*H x N x K
        k = k.view(len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # B x H x L x K -> B*H x L x K -> B*H x K x L
        pq = pq.view(bsz * self.num_heads, -1, self.head_dim).transpose(1, 2)
        # B*H x N x L
        pattn = k.bmm(pq)
        pattn = F.elu(pattn) + 1.0
        return pattn

    def forward(
        self,
        query,
        pquery,
        key_padding_mask: Optional[Tensor] = None,
        pkey_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            pkey_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, proj_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        pq = None
        num_steps = None
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_pquery" in saved_state:
                # previous time steps are cached - no need to recompute pquery
                # B x H x L x K
                pq = saved_state["prev_pquery"]
            key_accum_mat = 0
            value_accum_mat = 0
        else:
            saved_state = None
            key_accum_mat = None
            value_accum_mat = None

        if pq is None:
            plen = pquery.size(0)
            # L x B x D -> L x B x H x K
            pq = self.pq_proj(pquery).view(plen, bsz, self.num_heads, self.head_dim)
            # L x B x H x K -> B x H x L x K
            pq = pq.permute(1, 2, 0, 3) * self.scaling

        plen = pq.size(2)
        # B*H x N x L
        pattn_weights = self._compute_pattention(pq, query, key_padding_mask)
        pattn_weights = self.dropout_module(pattn_weights)

        # N x B x D -> B*H x N x K
        q = self.q_proj(query) * self.scaling
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # N x B x D -> B*H x N x K
        if self.c_proj is not None:
            k = v = (
                self.c_proj(query)
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            k = (
                self.k_proj(query)
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            v = (
                self.v_proj(query)
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        efficient_causal_attention = (
            efficient_causal_attention_parallel
            if self.parallel
            else efficient_causal_attention_seq
        )

        if saved_state is not None:
            # key accumulative matrix are store with shape (bsz, num_heads, head_dim, plen)
            if "prev_key_accum_mat" in saved_state:
                _prev_key_accum_mat = saved_state["prev_key_accum_mat"]
                key_accum_mat = _prev_key_accum_mat.view(
                    bsz * self.num_heads, self.head_dim, plen
                )
            # value accumulative matrix are store with shape (bsz, num_heads, plen, head_dim)
            if "prev_value_accum_mat" in saved_state:
                _prev_value_accum_mat = saved_state["prev_value_accum_mat"]
                value_accum_mat = _prev_value_accum_mat.view(
                    bsz * self.num_heads, plen, self.head_dim
                )
            if "prev_num_steps" in saved_state:
                _prev_num_steps = saved_state["prev_num_steps"]
                num_steps = _prev_num_steps.view(bsz * self.num_heads) + 1.0

        if num_steps is None:
            num_steps = query.new_ones(bsz * self.num_heads)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if pkey_padding_mask is not None and pkey_padding_mask.dim() == 0:
            pkey_padding_mask = None

        if incremental_state is not None:
            attn_weights, key_accum_mat = incremental_causal_attention(
                q, k, pattn_weights, key_accum_mat, num_steps
            )
        else:
            attn_weights = efficient_causal_attention(q, k, pattn_weights)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, plen]

        if pkey_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, plen)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    pkey_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(
                    pkey_padding_mask, float("-inf")
                )
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, plen)

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        if incremental_state is not None:
            attn, value_accum_mat = incremental_causal_attention(
                attn_probs, pattn_weights, v, value_accum_mat, num_steps
            )
        else:
            attn = efficient_causal_attention(attn_probs, pattn_weights, v)

        if saved_state is not None:
            saved_state["prev_pquery"] = pq
            saved_state["prev_key_accum_mat"] = key_accum_mat.view(
                bsz, self.num_heads, self.head_dim, plen
            )
            saved_state["prev_value_accum_mat"] = value_accum_mat.view(
                bsz, self.num_heads, plen, self.head_dim
            )
            saved_state["prev_num_steps"] = num_steps.view(bsz, self.num_heads)
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, plen
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


def efficient_causal_attention_seq(x, y, z):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, n, d1)`
        y (Tensor): Tensor with shape `(batch, n, d1)`
        z (Tensor): Tensor with shape '(batch, n, d2)`
    return:
    """
    n = x.size(1)
    rets = []
    accum_mat = 0
    for i in range(n):
        xx = x[:, i : i + 1]  # B x 1 x d1
        yy = y[:, i : i + 1]  # B x 1 x d1
        zz = z[:, i : i + 1]  # B x 1 x d2

        # B x d1 x d2
        accum_mat = accum_mat + torch.bmm(yy.transpose(1, 2), zz)
        # B x 1 x d2
        rets.append(torch.bmm(xx, accum_mat).div(i + 1.0))
    # B x N x d2
    return torch.cat(rets, dim=1)


def efficient_causal_attention_parallel(x, y, z):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, n, d1)`
        y (Tensor): Tensor with shape `(batch, n, d1)`
        z (Tensor): Tensor with shape '(batch, n, d2)`
    return:
    """
    bsz, n, d1 = x.size()
    # (bsz, n, d1, 1) x (bsz, n, 1, d2) -> (bsz, n, d1, d2)
    sum_mat = torch.matmul(y.unsqueeze(3), z.unsqueeze(2))
    accum_mat = torch.cumsum(sum_mat, dim=1)
    # (bsz, n, 1, d1) x (bsz, n, d1, d2) -> (bsz, n, 1, d2) -> (bsz, n, d2)
    res = torch.matmul(x.unsqueeze(2), accum_mat).squeeze(2)
    # (1, n, 1)
    length_div = torch.arange(1, n + 1, device=x.device).unsqueeze(0).unsqueeze(2)
    res = res / length_div
    return res


def incremental_causal_attention(x, y, z, accum_mat, n):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, 1, d1)`
        y (Tensor): Tensor with shape `(batch, 1, d1)`
        z (Tensor): Tensor with shape `(batch, 1, d2)`
        accum_mat (Tensor): Tensor with shape `(batch, d1, d2)`
        n (Tensor): number of steps with shape `(batch, )`
    return:
    """
    bsz = n.size(0)
    # B x d1 x d2
    accum_mat = accum_mat + torch.bmm(y.transpose(1, 2), z)
    # B x 1 x d2
    out = torch.bmm(x, accum_mat).div(n.view(bsz, 1, 1))
    return out, accum_mat
