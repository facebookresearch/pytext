#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.vocab import ScriptVocabulary

from .beam_search import BeamSearch_UTTERANCE_EXP


@torch.jit.script
def get_single_unk_token(
    src_tokens: List[str], word_ids: List[int], copy_unk_token: bool, unk_idx: int
):
    """Returns the string representation of the first UNK
       we get in our source utterance. We can then use this string instead of
       writing "<UNK>" in our decoding.
    """
    if copy_unk_token:
        for i, x in enumerate(word_ids):
            if x == unk_idx:
                return src_tokens[i]
    return None


@torch.jit.script
def get_target_length(
    src_len: int,
    targetlen_cap: int,
    targetlen_a: float,
    targetlen_b: float,
    targetlen_c: float,
):
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


class Seq2SeqJIT_BASE:
    def __init__(
        self,
        src_dict,
        tgt_dict,
        model,
        beam_decode,
        targetlen_cap,
        targetlen_a,
        targetlen_b,
        targetlen_c,
        filter_eos_bos,
        copy_unk_token=False,
    ):
        self.source_vocab = ScriptVocabulary(
            src_dict._vocab,
            src_dict.get_unk_index(),
            bos_idx=src_dict.get_bos_index(),
            eos_idx=src_dict.get_eos_index(),
        )
        self.target_vocab = ScriptVocabulary(
            tgt_dict._vocab,
            tgt_dict.get_unk_index(),
            bos_idx=tgt_dict.get_unk_index(),
            eos_idx=tgt_dict.get_eos_index(),
        )
        self.targetlen_cap = torch.jit.Attribute(targetlen_cap, int)
        self.targetlen_a = torch.jit.Attribute(targetlen_a, float)
        self.targetlen_b = torch.jit.Attribute(targetlen_b, float)
        self.targetlen_c = torch.jit.Attribute(targetlen_c, float)
        self.filter_eos_bos = torch.jit.Attribute(filter_eos_bos, bool)
        self.model = model
        self.beam_decode = beam_decode

        self.copy_unk_token = torch.jit.Attribute(copy_unk_token, bool)
        self.unk_idx = torch.jit.Attribute(self.source_vocab.unk_idx, int)


class Seq2SeqJIT_UTTERANCE(torch.jit.ScriptModule):
    def __init__(
        self,
        src_dict,
        tgt_dict,
        model,
        beam_decode,
        targetlen_cap,
        targetlen_a,
        targetlen_b,
        targetlen_c,
        filter_eos_bos,
        copy_unk_token=False,
    ):
        super().__init__()
        Seq2SeqJIT_BASE.__init__(
            self,
            src_dict,
            tgt_dict,
            model,
            beam_decode,
            targetlen_cap,
            targetlen_a,
            targetlen_b,
            targetlen_c,
            filter_eos_bos,
            copy_unk_token,
        )
        assert isinstance(self.model, BeamSearch_UTTERANCE_EXP)

    @torch.jit.script_method
    def forward(
        self, src_tokens: List[str], src_len: int
    ) -> List[Tuple[List[str], float, List[float]]]:
        assert len(src_tokens) == src_len
        word_ids = self.source_vocab.lookup_indices_1d(src_tokens)

        # find, if there exists, the unk token in the source utterance.
        # If multiple we select the first unk token.
        single_unk_token = torch.jit.annotate(
            Optional[str],
            get_single_unk_token(
                src_tokens, word_ids, self.copy_unk_token, self.unk_idx
            ),
        )
        target_length = get_target_length(
            src_len,
            self.targetlen_cap,
            self.targetlen_a,
            self.targetlen_b,
            self.targetlen_c,
        )
        all_tokens, all_scores, all_weights, all_prev_indices = self.model(
            (torch.tensor(word_ids).reshape(-1, 1),),
            torch.tensor([src_len]),
            target_length,
        )

        hypos_etc = self.beam_decode(
            all_tokens, all_scores, all_weights, all_prev_indices, target_length
        )
        hypos_list = torch.jit.annotate(List[Tuple[List[str], float, List[float]]], [])

        filter_token_list = torch.jit.annotate(List[int], [])
        if self.filter_eos_bos:
            filter_token_list = [self.target_vocab.bos_idx, self.target_vocab.eos_idx]

        for seq in hypos_etc:
            hyopthesis = seq[0]
            stringified = self.target_vocab.lookup_words_1d(
                hyopthesis,
                filter_token_list=filter_token_list,
                possible_unk_token=single_unk_token,
            )
            hypos_list.append((stringified, seq[1], seq[2]))
        return hypos_list
