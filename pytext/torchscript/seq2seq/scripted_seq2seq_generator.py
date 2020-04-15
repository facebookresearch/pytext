#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple

import torch
from pytext.config import ConfigBase
from pytext.models.module import Module
from pytext.torchscript.seq2seq.beam_decode import BeamDecode
from pytext.torchscript.seq2seq.beam_search import BeamSearch


@torch.jit.script
def get_target_length(
    src_len: int,
    targetlen_cap: int,
    targetlen_a: float,
    targetlen_b: float,
    targetlen_c: float,
) -> int:
    target_length = int(
        min(
            targetlen_cap,
            src_len * targetlen_a * targetlen_a + src_len * targetlen_b + targetlen_c,
        )
    )
    assert target_length > 0, (
        "Target length cannot be less than 0 src_len:"
        + str(src_len)
        + " target_length:"
        + str(target_length)
    )
    return target_length


class ScriptedSequenceGenerator(Module):
    class Config(ConfigBase):
        beam_size: int = 2
        # We use a quardratic formula to generate the max target length
        #   min(targetlen_cap, targetlen_a*x^2 + targetlen_b*x + targetlen_c)
        targetlen_cap: int = 100
        targetlen_a: float = 0
        targetlen_b: float = 2
        targetlen_c: float = 2
        quantize: bool = True
        length_penalty: float = 0.25
        nbest: int = 2
        stop_at_eos: bool = True
        record_attention: bool = False

    @classmethod
    def from_config(cls, config, models, trg_dict_eos):
        return cls(models, trg_dict_eos, config)

    def __init__(self, models, trg_dict_eos, config):
        super().__init__()
        self.targetlen_cap = config.targetlen_cap
        self.targetlen_a: float = float(config.targetlen_a)
        self.targetlen_b: float = float(config.targetlen_b)
        self.targetlen_c: float = float(config.targetlen_c)

        self.beam_search = BeamSearch(
            models,
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

    def forward(
        self,
        src_tokens: torch.Tensor,
        dict_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        contextual_token_embedding: Optional[torch.Tensor],
        src_lengths: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, float, List[float], torch.Tensor, torch.Tensor]]:

        target_length = get_target_length(
            src_lengths.item(),
            self.targetlen_cap,
            self.targetlen_a,
            self.targetlen_b,
            self.targetlen_c,
        )

        all_tokens, all_scores, all_weights, all_prev_indices = self.beam_search(
            src_tokens,
            src_lengths,
            target_length,
            dict_feat,
            contextual_token_embedding,
        )

        return self.beam_decode(
            all_tokens, all_scores, all_weights, all_prev_indices, target_length
        )

    @torch.jit.export
    def generate_hypo(self, tensors: Dict[str, torch.Tensor]):

        actual_src_tokens = tensors["src_tokens"].t()
        dict_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        if "dict_tokens" in tensors:
            dict_feat = (
                tensors["dict_tokens"],
                tensors["dict_weights"],
                tensors["dict_lengths"],
            )

        contextual_token_embedding: Optional[torch.Tensor] = None
        if "contextual_token_embedding" in tensors:
            contextual_token_embedding = tensors["contextual_token_embedding"]

        hypos_etc = self.forward(
            actual_src_tokens,
            dict_feat,
            contextual_token_embedding,
            tensors["src_lengths"],
        )
        predictions = [[pred for pred, _, _, _, _ in hypos_etc]]
        scores = [[score for _, score, _, _, _ in hypos_etc]]
        return (predictions, scores)
