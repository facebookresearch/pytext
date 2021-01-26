#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.utils import (
    pad_2d,
    pad_2d_float,
    pad_3d_float,
    validate_padding_control,
)
from pytext.torchscript.vocab import ScriptVocabulary


class ScriptTensorizer(torch.jit.ScriptModule):
    device: str
    seq_padding_control: Optional[List[int]]
    batch_padding_control: Optional[List[int]]

    def __init__(self):
        super().__init__()
        self.device = ""
        self.seq_padding_control = torch.jit.Attribute(None, Optional[List[int]])
        self.batch_padding_control = torch.jit.Attribute(None, Optional[List[int]])

    @torch.jit.script_method
    def set_device(self, device: str):
        self.device = device

    @torch.jit.export
    def set_padding_control(self, dimension: str, padding_control: Optional[List[int]]):
        """
        This functions will be called to set a padding style.
        None - No padding
        List: first element 0, round seq length to the smallest list element larger than inputs
        """
        if not validate_padding_control(padding_control):
            raise RuntimeError("Malformed padding_control value")
        if dimension == "sequence_length":
            self.seq_padding_control = padding_control
        elif dimension == "batch_length":
            self.batch_padding_control = padding_control
        else:
            raise RuntimeError("Illegal padding dimension specified.")

    @torch.jit.script_method
    def tokenize(
        self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]
    ):
        """
        Process a single line of raw inputs into tokens, it supports
        two input formats:
            1) a single line of texts (single sentence or a pair)
            2) a single line of pre-processed tokens (single sentence or a pair)
        """
        raise NotImplementedError

    @torch.jit.script_method
    def numberize(
        self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]
    ):
        """
        Process a single line of raw inputs into numberized result, it supports
        two input formats:
            1) a single line of texts (single sentence or a pair)
            2) a single line of pre-processed tokens (single sentence or a pair)

        This function should handle the logic of calling tokenize(), add special
        tokens and vocab lookup.
        """
        raise NotImplementedError

    @torch.jit.script_method
    def tensorize(
        self,
        texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[List[str]]]] = None,
    ):
        """
        Process raw inputs into model input tensors, it supports two input
        formats:
            1) multiple rows of texts (single sentence or a pair)
            2) multiple rows of pre-processed tokens (single sentence or a pair)

        This function should handle the logic of calling numberize() and also
        padding the numberized result.
        """
        raise NotImplementedError

    @torch.jit.script_method
    def batch_size(
        self, texts: Optional[List[List[str]]], tokens: Optional[List[List[List[str]]]]
    ) -> int:
        if texts is not None:
            return len(texts)
        elif tokens is not None:
            return len(tokens)
        else:
            raise RuntimeError("Empty input for both texts and tokens.")

    @torch.jit.script_method
    def row_size(
        self,
        texts_list: Optional[List[List[str]]] = None,
        tokens_list: Optional[List[List[List[str]]]] = None,
    ) -> int:
        if texts_list is not None:
            return len(texts_list[0])
        elif tokens_list is not None:
            return len(tokens_list[0])
        else:
            raise RuntimeError("Empty input for both texts and tokens.")

    @torch.jit.script_method
    def get_texts_by_index(
        self, texts: Optional[List[List[str]]], index: int
    ) -> Optional[List[str]]:
        if texts is None:
            return None
        return texts[index]

    @torch.jit.script_method
    def get_tokens_by_index(
        self, tokens: Optional[List[List[List[str]]]], index: int
    ) -> Optional[List[List[str]]]:
        if tokens is None:
            return None
        return tokens[index]


class VocabLookup(torch.jit.ScriptModule):
    """
    TorchScript implementation of lookup_tokens() in pytext/data/tensorizers.py
    """

    def __init__(self, vocab: ScriptVocabulary):
        super().__init__()
        self.vocab = vocab

    @torch.jit.script_method
    def forward(
        self,
        tokens: List[Tuple[str, int, int]],
        bos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
        use_eos_token_for_bos: bool = False,
        max_seq_len: int = 2 ** 30,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Convert tokens into ids by doing vocab look-up.

        Convert tokens into ids by doing vocab look-up. It will also append
        bos & eos index into token_ids if needed. A token is represented by
        a Tuple[str, int, int], which is [token, start_index, end_index].

        Args:
            tokens: List of tokens with start and end position in the original
                text. start and end index could be optional (e.g value is -1)
            bos_idx: index of begin of sentence, optional.
            eos_idx: index of end of sentence, optional.
            use_eos_token_for_bos: use eos index as bos.
            max_seq_len: maximum tokens length.
        """
        # unwrap Optional typing
        if bos_idx is None:
            bos_idx = -1
        if eos_idx is None:
            eos_idx = -1

        text_tokens: List[str] = []
        start_idxs: List[int] = []
        end_idxs: List[int] = []

        max_seq_len = (
            max_seq_len - (1 if bos_idx >= 0 else 0) - (1 if eos_idx >= 0 else 0)
        )
        for i in range(min(len(tokens), max_seq_len)):
            token: Tuple[str, int, int] = tokens[i]
            text_tokens.append(token[0])
            start_idxs.append(token[1])
            end_idxs.append(token[2])

        # vocab lookup
        token_ids: List[int] = self.vocab.lookup_indices_1d(text_tokens)
        # add bos and eos index if needed
        if bos_idx >= 0:
            if use_eos_token_for_bos:
                bos_idx = eos_idx
            token_ids = [bos_idx] + token_ids
            start_idxs = [-1] + start_idxs
            end_idxs = [-1] + end_idxs
        if eos_idx >= 0:
            token_ids.append(eos_idx)
            start_idxs.append(-1)
            end_idxs.append(-1)
        return token_ids, start_idxs, end_idxs


class ScriptInteger1DListTensorizer(torch.jit.ScriptModule):
    """
    TorchScript implementation of Integer1DListTensorizer in pytext/data/tensorizers.py
    """

    def __init__(self):
        super().__init__()
        self.pad_idx = 0

    @torch.jit.script_method
    def numberize(self, integerList: List[int]) -> Tuple[List[int], int]:
        return integerList, len(integerList)

    @torch.jit.script_method
    def tensorize(
        self, integerList: List[List[int]], seq_lens: List[int]
    ) -> torch.Tensor:
        integerListTensor = torch.tensor(
            pad_2d(integerList, seq_lens=seq_lens, pad_idx=self.pad_idx),
            dtype=torch.long,
        )
        return integerListTensor

    @torch.jit.ignore
    def torchscriptify(self):
        return torch.jit.script(self)


class ScriptFloat1DListTensorizer(torch.jit.ScriptModule):
    """
    TorchScript implementation of Float1DListTensorizer in pytext/data/tensorizers.py
    """

    def __init__(self):
        super().__init__()
        self.pad_val = 1.0

    @torch.jit.script_method
    def numberize(self, floatList: List[float]) -> Tuple[List[float], int]:
        return floatList, len(floatList)

    @torch.jit.script_method
    def tensorize(
        self, floatLists: List[List[float]], seq_lens: List[int]
    ) -> torch.Tensor:
        floatListTensor = torch.tensor(
            pad_2d_float(floatLists, seq_lens=seq_lens, pad_val=self.pad_val),
            dtype=torch.float,
        )
        return floatListTensor

    def torchscriptify(self):
        return torch.jit.script(self)


class ScriptFloatListSeqTensorizer(torch.jit.ScriptModule):
    """
    TorchScript implementation of ScriptFloatListSeqTensorizer in pytext/data/tensorizers.py
    """

    def __init__(self, pad_token):
        super().__init__()
        self.pad_val = pad_token

    @torch.jit.script_method
    def numberize(self, floatList: List[List[float]]) -> Tuple[List[List[float]], int]:
        return (floatList, len(floatList))

    @torch.jit.script_method
    def tensorize(
        self, floatLists: List[List[List[float]]], seq_lens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        floatListTensor = torch.tensor(
            pad_3d_float(floatLists, seq_lens=seq_lens, pad_val=self.pad_val),
            dtype=torch.float,
        )
        seqLensTensor = torch.tensor(seq_lens, dtype=torch.long)
        return (floatListTensor, seqLensTensor)

    def torchscriptify(self):
        return torch.jit.script(self)
