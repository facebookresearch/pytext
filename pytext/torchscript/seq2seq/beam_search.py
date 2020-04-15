#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import List, Optional, Tuple

import torch
import torch.jit
import torch.jit.quantized
from torch import nn

from .decoder import DecoderBatchedStepEnsemble
from .encoder import EncoderEnsemble


@torch.jit.script
def get_first_decoder_step_input(
    beam_size: int = 5, eos_token_id: int = 0, src_length: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prev_tokens = torch.full([beam_size], eos_token_id, dtype=torch.long)
    prev_scores = torch.full([beam_size], 1, dtype=torch.float)
    prev_hypos = torch.full([beam_size], 0, dtype=torch.long)
    attention_weights = torch.full([beam_size, src_length], 1, dtype=torch.float)
    return prev_tokens, prev_scores, prev_hypos, attention_weights


class BeamSearch(nn.Module):
    def __init__(
        self,
        model_list,
        tgt_dict_eos,
        beam_size: int = 2,
        quantize: bool = False,
        record_attention: bool = False,
    ):
        super().__init__()
        self.models = model_list
        self.target_dict_eos = tgt_dict_eos
        self.beam_size = beam_size
        self.record_attention = record_attention

        # Script the encoder model
        encoder_ens = EncoderEnsemble(self.models, self.beam_size)
        if quantize:
            encoder_ens = torch.quantization.quantize_dynamic(
                encoder_ens,
                {torch.nn.Linear},  # Add after bug fix torch.nn.LSTM
                dtype=torch.qint8,
                inplace=False,
            )

        self.encoder_ens = torch.jit.script(encoder_ens)

        # Script the decoder step
        decoder_ens = DecoderBatchedStepEnsemble(
            self.models, beam_size, record_attention=record_attention
        )
        if quantize:
            decoder_ens = torch.quantization.quantize_dynamic(
                decoder_ens,
                {torch.nn.Linear},  # Add after bug fix torch.nn.LSTM
                dtype=torch.qint8,
                inplace=False,
            )

        self.decoder_ens = torch.jit.script(decoder_ens)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        num_steps: int,
        dict_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        contextual_token_embedding: Optional[torch.Tensor] = None,
    ):
        # Initialize incremental_states after every forward()
        self.decoder_ens.reset_incremental_states()

        # ARBABU: To do add an assert to make sure that batch size is 1
        decoder_ip = self.encoder_ens(
            src_tokens, src_lengths, dict_feat, contextual_token_embedding
        )

        (
            prev_token,
            prev_scores,
            prev_hypos_indices,
            attention_weights,
        ) = get_first_decoder_step_input(
            self.beam_size, self.target_dict_eos, src_lengths[0]
        )

        all_tokens_list = [prev_token]
        all_scores_list = [prev_scores]
        all_prev_indices_list = [prev_hypos_indices]
        all_attentions_list: List[torch.Tensor] = []
        if self.record_attention:
            all_attentions_list.append(attention_weights)

        for i in range(num_steps):
            (
                prev_token,
                prev_scores,
                prev_hypos_indices,
                attention_weights,
                decoder_ip,
            ) = self.decoder_ens(prev_token, prev_scores, i + 1, decoder_ip)
            all_tokens_list.append(prev_token)
            all_scores_list.append(prev_scores)
            all_prev_indices_list.append(prev_hypos_indices)
            if self.record_attention:
                all_attentions_list.append(attention_weights)

        all_tokens = torch.stack(all_tokens_list)
        all_scores = torch.stack(all_scores_list)
        all_prev_indices = torch.stack(all_prev_indices_list)
        if self.record_attention:
            all_attn_weights = torch.stack(all_attentions_list)
        else:
            all_attn_weights = torch.zeros(
                num_steps + 1, self.beam_size, src_tokens.size(1)
            )

        return all_tokens, all_scores, all_attn_weights, all_prev_indices
