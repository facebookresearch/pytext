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


def build_wordpiece_vocab(
    wordpiece_vocab: Dict, special_token_replacements: Dict[str, SpecialToken] = None
) -> Vocabulary:
    """
    Build the wordpiece vocab. This requires a special function since the vocab is
    extracted from the tokenizer.
    """
    return Vocabulary(
        [token for token, _ in wordpiece_vocab.items()],
        replacements=special_token_replacements,
    )


def build_fairseq_vocab(
    dictionary_class: Dictionary,
    vocab_file: str,
    special_token_replacements: Dict[str, SpecialToken] = None,
    max_vocab: int = -1,
    min_count: int = -1,
) -> Vocabulary:
    """
    Function builds a PyText vocabulary for models pre-trained using Fairseq
    modules. The dictionary class can take any Dictionary (or its child) class
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
        add_bos_token: bool = False
        add_eos_token: bool = True
        # If set to true, this will call the wrap function
        # which has logic for doing model specific wrapping of numberized
        # sentences. This flag also allows us to use the add_bos_token and
        # add_eos_token flags for what they were originally meant to do
        wrap_special_tokens: bool = True
        bos_token: str = "[CLS]"
        eos_token: str = "[SEP]"
        pad_token: str = "[PAD]"
        unk_token: str = "[UNK]"
        mask_token: str = "[MASK]"
        vocab_file: str = WordPieceTokenizer.Config().wordpiece_vocab_path

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        special_token_replacements = {
            config.unk_token: UNK,
            config.pad_token: PAD,
            config.bos_token: BOS,
            config.eos_token: EOS,
            config.mask_token: MASK,
        }
        # Build the Vocabulary. Here we need a special case for
        # WordPiece since the vocabulary for WordPiece is created inside the
        # tokenizer
        if isinstance(tokenizer, WordPieceTokenizer):
            vocab = build_wordpiece_vocab(tokenizer.vocab, special_token_replacements)
        else:
            vocab = build_fairseq_vocab(
                dictionary_class=BertDictionary,
                vocab_file=config.vocab_file,
                special_token_replacements=special_token_replacements,
            )

        return cls(
            columns=config.columns,
            tokenizer=tokenizer,
            vocab=vocab,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            use_eos_token_for_bos=config.use_eos_token_for_bos,
            wrap_special_tokens=config.wrap_special_tokens,
            max_seq_len=config.max_seq_len,
        )

    def __init__(
        self,
        columns: List[str],
        vocab: Vocabulary,
        tokenizer: Tokenizer,
        add_bos_token: bool = Config.add_bos_token,
        add_eos_token: bool = Config.add_eos_token,
        use_eos_token_for_bos: bool = Config.use_eos_token_for_bos,
        wrap_special_tokens: bool = Config.wrap_special_tokens,
        max_seq_len: int = Config.max_seq_len,
    ) -> None:
        super().__init__(
            text_column=None,
            vocab=vocab,
            tokenizer=tokenizer,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_eos_token_for_bos=use_eos_token_for_bos,
            max_seq_len=max_seq_len,
        )
        self.columns = columns
        self.wrap_special_tokens = wrap_special_tokens

    def initialize(self, vocab_builder=None, from_scratch=True):
        # vocab for BERT is already set
        return
        # we need yield here to make this function a generator
        yield

    @property
    def column_schema(self):
        return [(column, str) for column in self.columns]

    def _lookup_tokens(self, text):
        """
        lookup_tokens does both the numberization as well as the wrapping
        of text with special tokens (eg: EOS and BOS). In some cases (eg: BERT),
        we use the add_bos_token flag to add a single BOS token at the start of
        the text instead of at the start of every sentence. As a result the call
        to lookup_tokens can be model specific and each child of BERTTensorizer
        has the option of customizing this call.
        """
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=self.vocab.bos_token if self.add_bos_token else None,
            eos_token=self.vocab.eos_token if self.add_eos_token else None,
            max_seq_len=self.max_seq_len,
        )

    def wrap(self, sentences: List[List[str]]):
        """
        Function which handles wrapping of numberized text with special tokens.
        """
        bos_token = (
            self.vocab.eos_token if self.use_eos_token_for_bos else self.vocab.bos_token
        )
        sentences[0] = [self.vocab.idx[bos_token]] + sentences[0]
        return sentences

    def numberize(self, row):
        """
        Tokenize and look up in vocabulary. This also adds a BOS token once at
        the beginning of the final numberized output.
        """
        sentences = [self._lookup_tokens(row[column])[0] for column in self.columns]
        if self.wrap_special_tokens:
            sentences = self.wrap(sentences)
        seq_lens = (len(sentence) for sentence in sentences)
        segment_labels = ([i] * seq_len for i, seq_len in enumerate(seq_lens))
        tokens = list(itertools.chain(*sentences))
        segment_labels = list(itertools.chain(*segment_labels))
        seq_len = len(tokens)
        positions = [index for index in range(seq_len)]
        # tokens, segment_label, seq_len
        return tokens, segment_labels, seq_len, positions

    def sort_key(self, row):
        return row[2]

    def tensorize(self, batch):
        tokens, segment_labels, seq_lens, positions = zip(*batch)
        tokens = pad_and_tensorize(tokens, self.vocab.get_pad_index())
        pad_mask = (tokens != self.vocab.get_pad_index()).long()
        segment_labels = pad_and_tensorize(segment_labels, self.vocab.get_pad_index())
        positions = pad_and_tensorize(positions)
        return tokens, pad_mask, segment_labels, positions
