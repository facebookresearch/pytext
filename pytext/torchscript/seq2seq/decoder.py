#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Tuple

import torch
import torch.jit
import torch.jit.quantized
from torch import Tensor, nn


class DecoderBatchedStepEnsemble(nn.Module):
    """
    This method should have a common interface such that it can be called after
    the encoder as well as after the decoder
    """

    incremental_states: Dict[str, Dict[str, Tensor]]

    def __init__(self, models, beam_size, record_attention=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.incremental_states = {}
        self.beam_size = beam_size
        self.record_attention = record_attention

    @torch.jit.export
    def reset_incremental_states(self):
        for idx, _model in enumerate(self.models):
            self.incremental_states[str(idx)] = {}

    def forward(
        self,
        prev_tokens: Tensor,
        prev_scores: Tensor,
        timestep: int,
        decoder_ips: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]:
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        HOWEVER: after the first step, encoder outputs (i.e, the first
        len(self.models) elements of inputs) must be tiled k (beam size)
        times on the batch dimension (axis 1).
        """

        # from flat to (batch x 1)
        prev_tokens = prev_tokens.unsqueeze(1)

        log_probs_per_model = torch.jit.annotate(List[Tensor], [])
        attn_weights_per_model = torch.jit.annotate(List[Tensor], [])

        futures = torch.jit.annotate(List[Tuple[Tensor, Dict[str, Tensor]]], [])
        for idx, model in enumerate(self.models):
            decoder_ip = decoder_ips[idx]
            incremental_state = self.incremental_states[str(idx)]
            # jit.fork() once https://github.com/pytorch/pytorch/issues/26578
            # lands
            fut = model.decoder(
                prev_tokens,
                decoder_ip,
                incremental_state=incremental_state,
                timestep=timestep,
            )
            futures.append(fut)

        # We do this in two separate loops only to permit model level parallelism
        # We fork in the first loop and wait in the second
        for idx, _model in enumerate(self.models):
            # jit.wait()
            fut = futures[idx]
            log_probs, features = fut
            log_probs_per_model.append(log_probs)
            if "attn_scores" in features:
                attn_weights_per_model.append(features["attn_scores"])

        (
            best_scores,
            best_tokens,
            prev_hypos,
            attention_weights,
        ) = self.beam_search_aggregate_topk(
            log_probs_per_model,
            attn_weights_per_model,
            prev_scores,
            self.beam_size,
            self.record_attention,
        )

        for model_state_ptr, model in enumerate(self.models):
            incremental_state = self.incremental_states[str(model_state_ptr)]
            model.decoder.reorder_incremental_state(incremental_state, prev_hypos)

        return best_tokens, best_scores, prev_hypos, attention_weights, decoder_ips

    def beam_search_aggregate_topk(
        self,
        log_probs_per_model: List[torch.Tensor],
        attn_weights_per_model: List[torch.Tensor],
        prev_scores: torch.Tensor,
        beam_size: int,
        record_attention: bool,
    ):
        average_log_probs = torch.mean(
            torch.cat(log_probs_per_model, dim=1), dim=1, keepdim=True
        )

        best_scores_k_by_k, best_tokens_k_by_k = torch.topk(
            average_log_probs.squeeze(1), k=beam_size
        )

        prev_scores_k_by_k = prev_scores.view(-1, 1).expand(-1, beam_size)
        total_scores_k_by_k = best_scores_k_by_k + prev_scores_k_by_k

        # flatten to take top k over all (beam x beam) hypos
        total_scores_flat = total_scores_k_by_k.view(-1)
        best_tokens_flat = best_tokens_k_by_k.view(-1)

        best_scores, best_indices = torch.topk(total_scores_flat, k=beam_size)

        best_tokens = best_tokens_flat.index_select(dim=0, index=best_indices).view(-1)

        # integer division to determine which input produced each successor
        prev_hypos = best_indices // beam_size

        if record_attention:
            average_attn_weights = torch.mean(
                torch.cat(attn_weights_per_model, dim=1), dim=1, keepdim=True
            )
            attention_weights = average_attn_weights.index_select(
                dim=0, index=prev_hypos
            )
            attention_weights = attention_weights.squeeze_(1)
        else:
            attention_weights = torch.zeros(
                beam_size, attn_weights_per_model[0].size(2)
            )

        return (best_scores, best_tokens, prev_hypos, attention_weights)
