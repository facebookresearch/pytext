#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import Dict, List

from fairseq.data.dictionary import Dictionary
from fairseq.data.legacy.masked_lm_dictionary import BertDictionary
from pytext.config.component import ComponentType, create_component
from pytext.data.tensorizers import TokenTensorizer, lookup_tokens
from pytext.data.tokenizers import Tokenizer, WordPieceTokenizer
from pytext.data.utils import (
    BOS,
    EOS,
    MASK,
    PAD,
    UNK,
    SpecialToken,
    Vocabulary,
    pad_and_tensorize,
)
from pytext.torchscript.tensorizer import ScriptBERTTensorizer
from pytext.torchscript.vocab import ScriptVocabulary


def build_fairseq_vocab(
    vocab_file: str,
    dictionary_class: Dictionary = Dictionary,
    special_token_replacements: Dict[str, SpecialToken] = None,
    max_vocab: int = -1,
    min_count: int = -1,
) -> Vocabulary:
    """
    Function builds a PyText vocabulary for models pre-trained using Fairseq
    modules. The dictionary class can take any Fairseq Dictionary class
    and is used to load the vocab file.
    """
    dictionary = dictionary_class.load(vocab_file)
    # finalize will sort the dict based on frequency so only do this if
    # a min_count or max_vocab size is specified
    if min_count > 0 or max_vocab > 0:
        dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
    return Vocabulary(
        dictionary.symbols, dictionary.count, replacements=special_token_replacements
    )


class BERTTensorizer(TokenTensorizer):
    """
    Tensorizer for BERT tasks.  Works for single sentence, sentence pair, triples etc.
    """

    __EXPANSIBLE__ = True

    class Config(TokenTensorizer.Config):
        #: The tokenizer to use to split input text into tokens.
        columns: List[str] = ["text"]
        tokenizer: Tokenizer.Config = WordPieceTokenizer.Config()
        add_bos_token: bool = True
        add_eos_token: bool = True
        bos_token: str = "[CLS]"
        eos_token: str = "[SEP]"
        pad_token: str = "[PAD]"
        unk_token: str = "[UNK]"
        mask_token: str = "[MASK]"
        vocab_file: str = WordPieceTokenizer.Config().wordpiece_vocab_path

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        special_token_replacements = {
            config.unk_token: UNK,
            config.pad_token: PAD,
            config.bos_token: BOS,
            config.eos_token: EOS,
            config.mask_token: MASK,
        }
        if isinstance(tokenizer, WordPieceTokenizer):
            vocab = Vocabulary(
                [token for token, _ in tokenizer.vocab.items()],
                replacements=special_token_replacements,
            )
        else:
            vocab = build_fairseq_vocab(
                dictionary_class=BertDictionary,
                vocab_file=config.vocab_file,
                special_token_replacements=special_token_replacements,
            )
        return cls(
            columns=config.columns,
            tokenizer=tokenizer,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            use_eos_token_for_bos=config.use_eos_token_for_bos,
            max_seq_len=config.max_seq_len,
            vocab=vocab,
            **kwargs,
        )

    def __init__(self, columns, **kwargs):
        super().__init__(text_column=None, **kwargs)
        self.columns = columns
        # Manually initialize column_schema since we are sending None to TokenTensorizer

    def initialize(self, vocab_builder=None, from_scratch=True):
        # vocab for BERT is already set
        return
        # we need yield here to make this function a generator
        yield

    @property
    def column_schema(self):
        return [(column, str) for column in self.columns]

    def _lookup_tokens(self, text):
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=None,
            eos_token=self.vocab.eos_token,
            max_seq_len=self.max_seq_len,
        )

    def numberize(self, row):
        """Tokenize, look up in vocabulary."""
        sentences = [self._lookup_tokens(row[column])[0] for column in self.columns]
        if self.add_bos_token:
            bos_token = (
                self.vocab.eos_token
                if self.use_eos_token_for_bos
                else self.vocab.bos_token
            )
            sentences[0] = [self.vocab.idx[bos_token]] + sentences[0]
        seq_lens = (len(sentence) for sentence in sentences)
        segment_labels = ([i] * seq_len for i, seq_len in enumerate(seq_lens))
        tokens = list(itertools.chain(*sentences))
        segment_labels = list(itertools.chain(*segment_labels))
        seq_len = len(tokens)
        # tokens, segment_label, seq_len
        return tokens, segment_labels, seq_len

    def sort_key(self, row):
        return row[2]

    def tensorize(self, batch):
        tokens, segment_labels, seq_lens = zip(*batch)
        tokens = pad_and_tensorize(tokens, self.vocab.get_pad_index())
        pad_mask = (tokens != self.vocab.get_pad_index()).long()
        segment_labels = pad_and_tensorize(segment_labels, self.vocab.get_pad_index())
        return tokens, pad_mask, segment_labels

    def torchscriptify(self):
        return ScriptBERTTensorizer(
            tokenizer=self.tokenizer.torchscriptify(),
            vocab=ScriptVocabulary(
                list(self.vocab),
                pad_idx=self.vocab.get_pad_index(),
                bos_idx=self.vocab.get_bos_index(),
                eos_idx=self.vocab.get_eos_index(),
            ),
            max_seq_len=self.max_seq_len,
            wrap_special_tokens=True,
        )
