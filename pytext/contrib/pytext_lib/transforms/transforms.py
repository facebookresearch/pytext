#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from fairseq.data.dictionary import Dictionary
from pytext.common.constants import SpecialTokens
from pytext.data.bert_tensorizer import build_fairseq_vocab
from pytext.data.utils import SpecialToken, VocabBuilder, Vocabulary
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


class Transform(nn.Module):
    @property
    def is_jitable(self) -> bool:
        return True


class Tokens(NamedTuple):
    token_texts: List[str]
    start_ids: List[int]
    end_ids: List[int]


class TokenizerTransform(Transform):
    def __init__(self, tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, text: str) -> Tokens:
        token_texts: List[str] = []
        start_ids: List[int] = []
        end_ids: List[int] = []
        for token in self.tokenizer.tokenize(text):
            if isinstance(token, str):
                token_texts.append(token)
            elif isinstance(token, (list, tuple)):
                token_texts.append(token[0])
                if len(token) >= 3:
                    start_ids.append(token[1])
                    end_ids.append(token[2])
            else:
                raise TypeError(
                    f"invalid token type {type(token)} returned from {self.tokenizer}"
                )
        if len(start_ids) == 0:
            start_ids = [-1] * len(token_texts)
        if len(end_ids) == 0:
            end_ids = [-1] * len(token_texts)
        return Tokens(token_texts=token_texts, start_ids=start_ids, end_ids=end_ids)

    @property
    def is_jitable(self) -> bool:
        return True


class WhiteSpaceTokenizer(nn.Module):
    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def forward(self, text: str) -> List[str]:
        return self.tokenize(text)


class WhitespaceTokenizerTransform(TokenizerTransform):
    def __init__(self, sp_model_path: Optional[str] = None):
        super().__init__(tokenizer=WhiteSpaceTokenizer())

    @property
    def is_jitable(self) -> bool:
        return False


class VocabTransform(Transform):
    def __init__(self, vocab: Vocabulary):
        super().__init__()
        self.vocab = ScriptVocabulary(
            list(vocab),
            pad_idx=vocab.get_pad_index(-1),
            bos_idx=vocab.get_bos_index(-1),
            eos_idx=vocab.get_eos_index(-1),
            unk_idx=vocab.get_unk_index(-1),
        )

    def forward(self, tokens: Tokens) -> Dict[str, torch.Tensor]:
        token_ids: List[int] = self.vocab.lookup_indices_1d(tokens.token_texts)
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "start_ids": torch.tensor(tokens.start_ids, dtype=torch.long),
            "end_ids": torch.tensor(tokens.end_ids, dtype=torch.long),
        }

    @property
    def is_jitable(self) -> bool:
        return True


class TruncateTransform(Transform):
    def __init__(self, bos_idx: int = -1, eos_idx: int = -1, max_seq_len: int = 256):
        super().__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_ids = tokens["token_ids"]
        start_ids = tokens["start_ids"]
        end_ids = tokens["end_ids"]
        max_seq_len = (
            self.max_seq_len
            - (1 if self.bos_idx >= 0 else 0)
            - (1 if self.eos_idx >= 0 else 0)
        )
        if len(token_ids) > max_seq_len:
            token_ids = torch.narrow(token_ids, 0, 0, max_seq_len)
        if len(start_ids) > max_seq_len:
            start_ids = torch.narrow(start_ids, 0, 0, max_seq_len)
        if len(end_ids) > max_seq_len:
            end_ids = torch.narrow(end_ids, 0, 0, max_seq_len)
        # add bos and eos index if needed
        if self.bos_idx >= 0:
            token_ids = torch.cat((torch.tensor([self.bos_idx]), token_ids))
            start_ids = torch.cat((torch.tensor([-1]), start_ids))
            end_ids = torch.cat((torch.tensor([-1]), end_ids))
        if self.eos_idx >= 0:
            token_ids = torch.cat((token_ids, torch.tensor([self.eos_idx])))
            start_ids = torch.cat((start_ids, torch.tensor([-1])))
            end_ids = torch.cat((end_ids, torch.tensor([-1])))
        return {"token_ids": token_ids, "start_ids": start_ids, "end_ids": end_ids}

    @property
    def is_jitable(self) -> bool:
        return True


class LabelTransform(Transform):
    def __init__(self, label_names: List[str]):
        super().__init__()
        self.vocab = Vocabulary(label_names)

    def forward(self, label: str) -> Dict[str, torch.Tensor]:
        label_id = self.vocab.lookup_all(label)
        return {"label_ids": torch.tensor(label_id, dtype=torch.long)}

    @property
    def is_jitable(self) -> bool:
        return False


def build_vocab(vocab_file):
    vocab_builder = VocabBuilder()
    with PathManager.open(vocab_file) as f:
        vocab_builder.add_from_file(
            f, skip_header_line=False, lowercase_tokens=False, size=0
        )
    return vocab_builder.make_vocab()


def build_fairseq_vocab(
    vocab_file: str,
    dictionary_class: Dictionary = Dictionary,
    special_token_replacements: Dict[str, SpecialToken] = None,
    max_vocab: int = -1,
    min_count: int = -1,
    tokens_to_add: Optional[List[str]] = None,
):
    """
    Function builds a PyText vocabulary for models pre-trained using Fairseq
    modules. The dictionary class can take any Fairseq Dictionary class
    and is used to load the vocab file.
    """
    if not special_token_replacements:
        special_token_replacements = {
            "<pad>": SpecialTokens.PAD,
            "<s>": SpecialTokens.BOS,
            "</s>": SpecialTokens.EOS,
            "<unk>": SpecialTokens.UNK,
            "<mask>": SpecialTokens.MASK,
        }
    with PathManager.open(vocab_file) as f:
        dictionary = dictionary_class.load(f)
        # finalize will sort the dict based on frequency so only do this if
        # a min_count or max_vocab size is specified
        if min_count > 0 or max_vocab > 0:
            dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
        if tokens_to_add:
            for token in tokens_to_add:
                dictionary.add_symbol(token)
        return Vocabulary(
            dictionary.symbols,
            dictionary.count,
            replacements=special_token_replacements,
        )
