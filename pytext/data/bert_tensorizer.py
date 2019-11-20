#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.data.legacy.masked_lm_dictionary import BertDictionary
from pytext.config.component import ComponentType, create_component
from pytext.data.tensorizers import Tensorizer, lookup_tokens
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
from pytext.utils.file_io import PathManager


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


class BERTTensorizerBase(Tensorizer):
    """
    Base Tensorizer class for all BERT style models including XLM,
    RoBERTa and XLM-R.
    """

    __EXPANSIBLE__ = True

    class Config(Tensorizer.Config):
        # BERT style models support multiple text inputs
        columns: List[str] = ["text"]
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        # base token-level tokenizer for sequence labeling tasks
        base_tokenizer: Optional[Tokenizer.Config] = None
        vocab_file: str = ""
        max_seq_len: int = 256

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Vocabulary = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = Config.max_seq_len,
        base_tokenizer: Tokenizer = None,
    ) -> None:
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.base_tokenizer = base_tokenizer
        self.max_seq_len = max_seq_len
        # Needed to ensure that we're not masking special tokens. By default
        # we use the BOS token from the vocab. If a class has different
        # behavior (eg: XLM), it needs to override this.
        self.bos_token = self.vocab.bos_token

    @property
    def column_schema(self):
        return [(column, str) for column in self.columns]

    def _lookup_tokens(self, text: str, seq_len: int = None):
        """
        This function knows how to call lookup_tokens with the correct
        settings for this model. The default behavior is to wrap the
        numberized text with distinct BOS and EOS tokens. The resulting
        vector would look something like this:
            [BOS, token1_id, . . . tokenN_id, EOS]

        The function also takes an optional seq_len parameter which is
        used to customize truncation in case we have multiple text fields.
        By default max_seq_len is used. It's upto the numberize function of
        the class to decide how to use the seq_len param.

        For example:
        - In the case of sentence pair classification, we might want both
        pieces of text have the same length which is half of the
        max_seq_len supported by the model.
        - In the case of QA, we might want to truncate the context by a
        seq_len which is longer than what we use for the question.
        """
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=self.vocab.bos_token,
            eos_token=self.vocab.eos_token,
            max_seq_len=seq_len if seq_len else self.max_seq_len,
        )

    def _wrap_numberized_text(
        self, numberized_sentences: List[List[str]]
    ) -> List[List[str]]:
        """
        If a class has a non-standard way of generating the final numberized text
        (eg: BERT) then a class specific version of wrap_numberized_text function
        should be implemented. This allows us to share the numberize
        function across classes without having to copy paste code. The default
        implementation doesnt do anything.
        """
        return numberized_sentences

    def numberize(self, row: Dict) -> Tuple[Any, ...]:
        """
        This function contains logic for converting tokens into ids based on
        the specified vocab. It also outputs, for each instance, the vectors
        needed to run the actual model.
        """
        sentences = [self._lookup_tokens(row[column])[0] for column in self.columns]
        sentences = self._wrap_numberized_text(sentences)
        seq_lens = (len(sentence) for sentence in sentences)
        segment_labels = ([i] * seq_len for i, seq_len in enumerate(seq_lens))
        tokens = list(itertools.chain(*sentences))
        segment_labels = list(itertools.chain(*segment_labels))
        seq_len = len(tokens)
        positions = list(range(seq_len))
        # tokens, segment_label, seq_len
        return tokens, segment_labels, seq_len, positions

    def tensorize(self, batch) -> Tuple[torch.Tensor, ...]:
        """
        Convert instance level vectors into batch level tensors.
        """
        tokens, segment_labels, seq_lens, positions = zip(*batch)
        tokens = pad_and_tensorize(tokens, self.vocab.get_pad_index())
        pad_mask = (tokens != self.vocab.get_pad_index()).long()
        segment_labels = pad_and_tensorize(segment_labels)
        positions = pad_and_tensorize(positions)
        return tokens, pad_mask, segment_labels, positions

    def initialize(self, vocab_builder=None, from_scratch=True):
        # vocab for BERT is already set
        return
        # we need yield here to make this function a generator
        yield

    def sort_key(self, row):
        return row[2]


class BERTTensorizer(BERTTensorizerBase):
    """
    Tensorizer for BERT tasks.  Works for single sentence, sentence pair, triples etc.
    """

    __EXPANSIBLE__ = True

    class Config(BERTTensorizerBase.Config):
        tokenizer: Tokenizer.Config = WordPieceTokenizer.Config()
        vocab_file: str = WordPieceTokenizer.Config().wordpiece_vocab_path

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        """
        from_config parses the config associated with the tensorizer and
        creates both the tokenizer and the Vocabulary object. The extra arguments
        passed as kwargs allow us to reuse thie function with variable number
        of arguments (eg: for classes which derive from this class).
        """
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        special_token_replacements = {
            "[UNK]": UNK,
            "[PAD]": PAD,
            "[CLS]": BOS,
            "[MASK]": MASK,
            "[SEP]": EOS,
        }
        if isinstance(tokenizer, WordPieceTokenizer):
            vocab = Vocabulary(
                [token for token, _ in tokenizer.vocab.items()],
                replacements=special_token_replacements,
            )
        else:
            with PathManager.open(config.vocab_file) as file_path:
                vocab = build_fairseq_vocab(
                    dictionary_class=BertDictionary,
                    vocab_file=file_path,
                    special_token_replacements=special_token_replacements,
                )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            **kwargs,
        )

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Vocabulary = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = Config.max_seq_len,
        **kwargs,
    ) -> None:
        super().__init__(
            columns=columns, vocab=vocab, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def _lookup_tokens(self, text: str, seq_len: int = None):
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=None,
            eos_token=self.vocab.eos_token,
            max_seq_len=seq_len if seq_len else self.max_seq_len,
        )

    def _wrap_numberized_text(
        self, numberized_sentences: List[List[str]]
    ) -> List[List[str]]:
        numberized_sentences[0] = [self.vocab.get_bos_index()] + numberized_sentences[0]
        return numberized_sentences

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
        )
