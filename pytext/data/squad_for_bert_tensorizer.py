#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import List

from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizer, build_fairseq_vocab
from pytext.data.roberta_tensorizer import RoBERTaTensorizer
from pytext.data.tokenizers import Tokenizer
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK, Vocabulary, pad_and_tensorize
from pytext.torchscript.tensorizer import ScriptRoBERTaTensorizerWithIndices
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


class SquadForBERTTensorizer(BERTTensorizer):
    """Produces BERT inputs and answer spans for Squad."""

    SPAN_PAD_IDX = -100

    class Config(BERTTensorizer.Config):
        columns: List[str] = ["question", "doc"]
        # for labels
        answers_column: str = "answers"
        answer_starts_column: str = "answer_starts"
        max_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config):
        # reuse parent class's from_config, which will pass extra args
        # in **kwargs to cls.__init__
        return super().from_config(
            config,
            answers_column=config.answers_column,
            answer_starts_column=config.answer_starts_column,
        )

    def __init__(
        self,
        answers_column: str = Config.answers_column,
        answer_starts_column: str = Config.answer_starts_column,
        **kwargs,
    ):
        # Arguments which are common to both current and base class are passed
        # as **kwargs. These are then passed to the __init__ of the base class
        super().__init__(**kwargs)
        self.answers_column = answers_column
        self.answer_starts_column = answer_starts_column

    def numberize(self, row):
        question_column, doc_column = self.columns
        doc_tokens, start_idx, end_idx = self._lookup_tokens(row[doc_column])
        question_tokens, _, _ = self._lookup_tokens(row[question_column])
        question_tokens = [self.vocab.get_bos_index()] + question_tokens
        seq_lens = (len(question_tokens), len(doc_tokens))
        segment_labels = ([i] * seq_len for i, seq_len in enumerate(seq_lens))
        tokens = list(itertools.chain(question_tokens, doc_tokens))
        segment_labels = list(itertools.chain(*segment_labels))
        seq_len = len(tokens)
        positions = list(range(seq_len))

        # now map original answer spans to tokenized spans
        offset = len(question_tokens)
        start_idx_map = {}
        end_idx_map = {}
        for tokenized_idx, (raw_start_idx, raw_end_idx) in enumerate(
            zip(start_idx[:-1], end_idx[:-1])
        ):
            start_idx_map[raw_start_idx] = tokenized_idx + offset
            end_idx_map[raw_end_idx] = tokenized_idx + offset

        answer_start_indices = [
            start_idx_map.get(raw_idx, self.SPAN_PAD_IDX)
            for raw_idx in row[self.answer_starts_column]
        ]
        answer_end_indices = [
            end_idx_map.get(raw_idx + len(answer), self.SPAN_PAD_IDX)
            for raw_idx, answer in zip(
                row[self.answer_starts_column], row[self.answers_column]
            )
        ]
        if not answer_start_indices and answer_end_indices:
            answer_start_indices = [self.SPAN_PAD_IDX]
            answer_end_indices = [self.SPAN_PAD_IDX]

        return (
            tokens,
            segment_labels,
            seq_len,
            positions,
            answer_start_indices,
            answer_end_indices,
        )

    def tensorize(self, batch):
        (
            tokens,
            segment_labels,
            seq_len,
            positions,
            answer_start_idx,
            answer_end_idx,
        ) = zip(*batch)
        tokens = pad_and_tensorize(tokens, self.vocab.get_pad_index())
        segment_labels = pad_and_tensorize(segment_labels, self.vocab.get_pad_index())
        pad_mask = (tokens != self.vocab.get_pad_index()).long()
        positions = pad_and_tensorize(positions)
        answer_start_idx = pad_and_tensorize(answer_start_idx, self.SPAN_PAD_IDX)
        answer_end_idx = pad_and_tensorize(answer_end_idx, self.SPAN_PAD_IDX)
        return (
            tokens,
            pad_mask,
            segment_labels,
            positions,
            answer_start_idx,
            answer_end_idx,
        )


class SquadForRoBERTaTensorizer(SquadForBERTTensorizer, RoBERTaTensorizer):
    """Produces RoBERTa inputs and answer spans for Squad."""

    class Config(RoBERTaTensorizer.Config):
        columns: List[str] = ["question", "doc"]
        # for labels
        answers_column: str = "answers"
        answer_starts_column: str = "answer_starts"
        max_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        with PathManager.open(config.vocab_file) as file_path:
            vocab = build_fairseq_vocab(
                vocab_file=file_path,
                special_token_replacements={
                    "<pad>": PAD,
                    "<s>": BOS,
                    "</s>": EOS,
                    "<unk>": UNK,
                    "<mask>": MASK,
                },
            )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            answers_column=config.answers_column,
            answer_starts_column=config.answer_starts_column,
        )

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Vocabulary = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = Config.max_seq_len,
        answers_column: str = Config.answers_column,
        answer_starts_column: str = Config.answer_starts_column,
    ):
        RoBERTaTensorizer.__init__(
            self,
            columns=columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        self.answers_column = answers_column
        self.answer_starts_column = answer_starts_column
        self.wrap_special_tokens = False

    def _lookup_tokens(self, text):
        return RoBERTaTensorizer._lookup_tokens(self, text)

    def torchscriptify(self):
        return ScriptRoBERTaTensorizerWithIndices(
            tokenizer=self.tokenizer.torchscriptify(),
            vocab=ScriptVocabulary(
                list(self.vocab),
                pad_idx=self.vocab.get_pad_index(),
                bos_idx=self.vocab.get_bos_index(),
                eos_idx=self.vocab.get_eos_index(),
            ),
            max_seq_len=self.max_seq_len,
        )
