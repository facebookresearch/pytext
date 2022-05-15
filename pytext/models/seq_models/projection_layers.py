#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pytext.models.seq_models.utils import Linear
from torch import nn, Tensor

from .attention import DecoupledMultiheadAttention
from .utils import verify_encoder_out


class DecoderWithLinearOutputProjection(nn.Module):
    """Simple linear projection from the hidden vector to
    vocab.
    """

    def __init__(self, src_dict, dst_dict, out_embed_dim=512, *args, **kwargs):
        super().__init__()
        self.linear_projection = Linear(out_embed_dim, len(dst_dict))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.linear_projection.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear_projection.bias)

    def forward(
        self,
        encoder_out: Dict[str, Tensor],
        decoder_out: Tuple[Tensor, Dict[str, Tensor]],
        incremental_state: Optional[Dict[str, Tensor]] = None,
    ):
        x, others = decoder_out
        logits = self.linear_projection(x)
        return logits, others

    @torch.jit.export
    def get_probs(
        self, decoder_out: Tuple[Tensor, Dict[str, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits, output_dict = decoder_out
        # Any large reductions such as softmax should occur in
        # 32 bits
        probs = F.softmax(logits.float(), dim=-1).to(logits.dtype)
        max_probs, idx = probs.max(dim=-1)
        return idx, max_probs, probs


class DecoupledDecoderHead(nn.Module):
    fixed_generation_vocab_expanded = torch.jit.Final[Tensor]

    def __init__(
        self,
        src_dict,
        dst_dict,
        out_embed_dim=512,
        encoder_hidden_dim=None,
        pointer_attention_heads=1,
        fixed_generation_vocab=None,
        attention_dropout=0.2,
        model_output_logprob=True,
    ):
        super().__init__()
        self.linear_projection = nn.Linear(
            out_embed_dim,
            len(fixed_generation_vocab) if fixed_generation_vocab else len(dst_dict),
        )
        self.num_embeddings = len(dst_dict)
        self.pointer_projection = nn.Linear(encoder_hidden_dim, out_embed_dim)
        self.pointer_prob_map = nn.Linear(out_embed_dim + out_embed_dim, 1)
        self.pointer_attention = DecoupledMultiheadAttention(
            out_embed_dim,
            out_embed_dim,
            pointer_attention_heads,
            src_length_mask=False,
            dropout=attention_dropout,
        )
        self.fixed_vocab = not (fixed_generation_vocab is None)
        if self.fixed_vocab:
            assert isinstance(
                fixed_generation_vocab, list
            ), "List of indices is what is expected for fixed_generation_vocab"
            self.fixed_generation_vocab_expanded: Tensor = (
                torch.tensor(fixed_generation_vocab, dtype=torch.long)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            # make TorchScript happy
            self.fixed_generation_vocab_expanded: Tensor = torch.zeros([1])

        self.register_buffer(
            "fixed_generation_vocab_expanded_buffer",
            self.fixed_generation_vocab_expanded,
        )
        self.model_output_logprob = model_output_logprob

    def forward(
        self,
        encoder_out: Dict[str, Tensor],
        decoder_out: Tuple[Tensor, Dict[str, Tensor]],
        incremental_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Variables for Shape comments
        ----------------------------------
        B: Batch
        T_src: Length of source sequence
        T_trg: Length of target seuqence
        C: hidden dimension
        V_ont: Size of ontology vocabulary
        V_trg: Size of full target vocabulary
        """
        self.verify_encoder_out(encoder_out)

        # `encoder_outs`: T_src x B x C
        encoder_outs = encoder_out["encoder_out"]

        encoder_mask: Optional[Tensor] = None
        if "encoder_mask" in encoder_out:
            # `encoder_mask`: B x T_src
            encoder_mask = encoder_out["encoder_mask"]

        # `src_tokens`: B x T_src
        src_tokens = self.get_pointer_src_tokens(encoder_out)

        # The line below will have to be uncommented once ELMo/dictfeat is supported
        # src_tokens = pytorch_translate_utils.get_source_tokens_tensor(src_tokens)

        # `x`: B x T_trg x C
        x, output_dict = decoder_out
        output_dict["decoder_out"] = x

        # `logits`: B x T_trg x V_ontology
        logits = self.linear_projection(x)  # B x T_trg x V_ontology

        # compute softmax generation probability over the fixed vocabulary
        # softmax in 32 bits
        # `optional_fixed_logits` B x T_trg x V_ontology
        optional_fixed_logits = F.softmax(logits.float(), dim=2).to(logits.dtype)
        if self.fixed_vocab:
            # we know have to project to the full vocab size in order
            # to get the mixture of probability distributions right.

            # create the a zero matrix over the full vocabulary
            # `optional_fixed_logits_1`: B x T_trg x V_trg
            optional_fixed_logits_1 = torch.zeros(
                (logits.size(0), logits.size(1), self.num_embeddings),
                device=logits.device,
                dtype=logits.dtype,
            )

            # Expand the fixed vocabulary over the sequence
            # `fixed_expanded`: B x T_trg x V_ontology
            fixed_expanded = self.fixed_generation_vocab_expanded_buffer.repeat(
                logits.size(0), logits.size(1), 1
            )

            # Expand ontology scores to the zero matrix over the full vocabulary
            optional_fixed_logits_1.scatter_add_(
                2, fixed_expanded, optional_fixed_logits
            )
            optional_fixed_logits = optional_fixed_logits_1

        # `encoder_outs`: T_src x B x C
        encoder_outs = self.pointer_projection(encoder_outs)

        # `cur_src_attn`: T_trg x B x C
        # `calc_src_attn_scores`: T_src, T_trg, B
        cur_src_attn, calc_src_attn_scores = self.pointer_attention(
            x, encoder_outs, encoder_mask, False
        )
        cur_src_attn = cur_src_attn.to(
            device=optional_fixed_logits.device, dtype=optional_fixed_logits.dtype
        )
        calc_src_attn_scores = calc_src_attn_scores.to(
            device=optional_fixed_logits.device, dtype=optional_fixed_logits.dtype
        )

        # `cur_src_attn`: B x T_trg x C
        cur_src_attn = cur_src_attn.transpose(0, 1)

        # `calc_src_attn_scores`: B x T_trg x T_src
        calc_src_attn_scores = calc_src_attn_scores.transpose(0, 2)

        # compute generation probability per token
        # `prob`: B x T_trg x C
        prob = torch.sigmoid(self.pointer_prob_map(torch.cat([cur_src_attn, x], dim=2)))

        # create zero matrix over the full vocabulary for
        # the copy scores
        # `vocab_attn_scores`: B x T_trg x V_trg
        vocab_attn_scores = torch.zeros(
            optional_fixed_logits.size(0),
            optional_fixed_logits.size(1),
            optional_fixed_logits.size(2),
            device=optional_fixed_logits.device,
            dtype=optional_fixed_logits.dtype,
        )

        # expand source tokens over the target sequence
        # `src_tokens_expanded`: B x T_trg x T_src
        src_tokens_expanded = src_tokens.unsqueeze(1).repeat(
            1, logits.size(1), 1
        )  # B x T_trg x T_src

        # calc_src_attn_scores are already probabilities

        # add copy probabilities to appropriate vocab indexes
        # in `vocab_attn_scores` matrix
        # `vocab_attn_scores`: B x T_trg x V_trg
        vocab_attn_scores.scatter_add_(2, src_tokens_expanded, calc_src_attn_scores)

        # Mix probabilities for copying tokens and generating tokens
        # `explicit_copy_probs`: B x T_trg x V_trg
        explicit_copy_probs = (
            prob * optional_fixed_logits + (1 - prob) * vocab_attn_scores
        )
        # full support occurs if not self.fixed_vocab otherwise not.
        # Taking log(0) = -inf which should not be an issue as long as the loss
        # does not touch it, which it shoudln't. If loss is nan then it's not a
        # straightforward copy task
        if self.model_output_logprob:
            model_out = (explicit_copy_probs + 1e-7).log()
        else:
            model_out = explicit_copy_probs + 1e-7
        return model_out, output_dict

    @torch.jit.export
    def get_probs(
        self, decoder_out: Tuple[Tensor, Dict[str, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        model_out, output_dict = decoder_out
        if self.model_output_logprob:
            probs = model_out.exp()
            max_probs, idx = probs.max(dim=-1)
        else:
            max_probs, idx = model_out.max(dim=-1)
        return idx, max_probs, model_out

    def get_pointer_src_tokens(self, encoder_out: Dict[str, Tensor]) -> torch.Tensor:
        return encoder_out["src_tokens"]

    def verify_encoder_out(self, encoder_out: Dict[str, Tensor]):
        verify_encoder_out(encoder_out, ["encoder_out", "src_tokens"])
