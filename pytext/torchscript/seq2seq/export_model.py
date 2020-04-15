#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.vocab import ScriptVocabulary


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


class Seq2SeqJIT(torch.nn.Module):
    def __init__(
        self,
        src_dict,
        tgt_dict,
        sequence_generator,
        filter_eos_bos,
        copy_unk_token=False,
        dictfeat_dict=None,
    ):
        super().__init__()
        self.source_vocab = ScriptVocabulary(
            src_dict._vocab,
            src_dict.get_unk_index(),
            bos_idx=src_dict.get_bos_index(-1),
            eos_idx=src_dict.get_eos_index(-1),
        )
        self.target_vocab = ScriptVocabulary(
            tgt_dict._vocab,
            tgt_dict.get_unk_index(),
            bos_idx=tgt_dict.get_bos_index(),
            eos_idx=tgt_dict.get_eos_index(),
        )
        if dictfeat_dict:
            self.dictfeat_vocab = ScriptVocabulary(
                dictfeat_dict._vocab,
                # We want to use the index for the source pad token
                pad_idx=dictfeat_dict.idx[src_dict[src_dict.get_pad_index()]],
            )
        else:
            # Optional types in Torchscript are a bit of a pain, so it's
            # more convenient to have an empty model than use None in
            # this case.
            self.dictfeat_vocab = ScriptVocabulary([])
        self.sequence_generator = sequence_generator

        self.copy_unk_token: bool = copy_unk_token
        self.unk_idx: int = self.source_vocab.unk_idx
        self.filter_eos_bos: bool = filter_eos_bos

    def prepare_generator_inputs(
        self,
        word_ids: List[int],
        dict_feat: Optional[Tuple[List[str], List[float], List[int]]] = None,
        contextual_token_embedding: Optional[List[float]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        src_len = len(word_ids)
        dict_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if dict_feat is not None:
            dict_tokens, dict_weights, dict_lengths = dict_feat
            dict_ids = self.dictfeat_vocab.lookup_indices_1d(dict_tokens)
            dict_tensors = (
                torch.tensor([dict_ids]),
                torch.tensor([dict_weights], dtype=torch.float),
                torch.tensor([dict_lengths]),
            )

        contextual_embedding_tensor: Optional[torch.Tensor] = None
        if contextual_token_embedding is not None:
            assert (
                len(contextual_token_embedding) % src_len == 0
                and len(contextual_token_embedding) > 0
            ), (
                f"Incorrect size for contextual embeddings: "
                f"{len(contextual_token_embedding)}, Expected a "
                f"non-zero multiple of input token count {src_len} "
            )
            contextual_embedding_tensor = torch.tensor(
                [contextual_token_embedding], dtype=torch.float
            )
        return (
            torch.tensor(word_ids).reshape(-1, 1),
            dict_tensors,
            contextual_embedding_tensor,
            torch.tensor([src_len]),
        )

    def forward(
        self,
        src_tokens: List[str],
        dict_feat: Optional[Tuple[List[str], List[float], List[int]]] = None,
        contextual_token_embedding: Optional[List[float]] = None,
    ) -> List[Tuple[List[str], float, List[float]]]:

        word_ids = self.source_vocab.lookup_indices_1d(src_tokens)

        # find, if there exists, the unk token in the source utterance.
        # If multiple we select the first unk token.
        single_unk_token: Optional[str] = get_single_unk_token(
            src_tokens, word_ids, self.copy_unk_token, self.unk_idx
        )

        (
            words,
            dict_tensors,
            contextual_embedding_tensor,
            src_lengths,
        ) = self.prepare_generator_inputs(
            word_ids, dict_feat, contextual_token_embedding
        )
        hypos_etc = self.sequence_generator(
            words, dict_tensors, contextual_embedding_tensor, src_lengths
        )
        hypos_list: List[Tuple[List[str], float, List[float]]] = []

        filter_token_list: List[int] = []
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
