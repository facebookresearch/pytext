#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Dict, List, Optional

from fairseq.data.dictionary import Dictionary
from pytext.common.constants import SpecialTokens
from pytext.data.bert_tensorizer import build_fairseq_vocab
from pytext.data.utils import SpecialToken, VocabBuilder, Vocabulary
from pytext.utils.file_io import PathManager


def build_vocab(vocab_file):
    vocab_builder = VocabBuilder()
    if vocab_file:
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
