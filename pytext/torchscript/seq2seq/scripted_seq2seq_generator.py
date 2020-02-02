#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict

import torch
from pytext.torchscript.seq2seq.beam_decode import BeamDecode
from pytext.torchscript.seq2seq.beam_search import BeamSearch_UTTERANCE_EXP
from pytext.torchscript.seq2seq.export_model import get_target_length


class ScriptedSequenceGenerator(torch.nn.Module):
    def __init__(self, models, trg_dict_eos, config):
        super().__init__()
        self.targetlen_cap = config.targetlen_cap
        self.targetlen_a = config.targetlen_a
        self.targetlen_b = config.targetlen_b
        self.targetlen_c: float = config.targetlen_c

        # Assume models are representative and get representative information
        # from the first model
        sample, check = models[0].get_example_and_check()

        self.beam_search = BeamSearch_UTTERANCE_EXP(
            [m.model for m in models],
            trg_dict_eos,
            beam_size=config.beam_size,
            quantize=config.quantize,
            record_attention=config.record_attention,
        )
        self.beam_decode = BeamDecode(
            eos_token_id=trg_dict_eos,
            length_penalty=config.length_penalty,
            nbest=config.nbest,
            beam_size=config.beam_size,
            stop_at_eos=config.stop_at_eos,
        )

    def generate_hypo(
        self, tensors: Dict[str, torch.Tensor], maxlen_a: float = 0.0, maxlen_b: int = 0
    ):
        src_tokens = tensors["src_tokens"]
        src_lengths = tensors["src_lengths"]
        actual_src_tokens = src_tokens.t()

        target_length = get_target_length(
            src_lengths.item(),
            self.targetlen_cap,
            maxlen_a,
            float(maxlen_b),
            float(self.targetlen_c),
        )
        all_tokens, all_scores, all_weights, all_prev_indices = self.beam_search(
            (actual_src_tokens,), src_lengths, target_length
        )

        hypos_etc = self.beam_decode(
            all_tokens, all_scores, all_weights, all_prev_indices, target_length
        )
        return [[pred for pred, _, _, _, _ in hypos_etc]]
