#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import List

import torch
from pytext import resources
from pytext.common.constants import SpecialTokens
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizer, build_fairseq_vocab
from pytext.data.roberta_tensorizer import RoBERTaTensorizer
from pytext.data.tensorizers import lookup_tokens
from pytext.data.tokenizers import Tokenizer
from pytext.data.utils import Vocabulary, pad_and_tensorize
from pytext.torchscript.tensorizer import ScriptRoBERTaTensorizerWithIndices
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


class SquadForBERTTensorizer(BERTTensorizer):
    """Produces BERT inputs and answer spans for Squad."""

    __EXPANSIBLE__ = True
    SPAN_PAD_IDX = -100

    class Config(BERTTensorizer.Config):
        columns: List[str] = ["question", "doc"]
        # for labels
        answers_column: str = "answers"
        answer_starts_column: str = "answer_starts"
        max_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        # reuse parent class's from_config, which will pass extra args
        # in **kwargs to cls.__init__
        return super().from_config(
            config,
            answers_column=config.answers_column,
            answer_starts_column=config.answer_starts_column,
            **kwargs,
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

    def _lookup_tokens(self, text: str, seq_len: int = None):
        # BoS token is added explicitly in numberize()
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=None,
            eos_token=self.vocab.eos_token,
            max_seq_len=seq_len if seq_len else self.max_seq_len,
        )

    def _calculate_answer_indices(self, row, offset, start_idx, end_idx):
        # now map original answer spans to tokenized spans
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
        if not (answer_start_indices and answer_end_indices):
            answer_start_indices = [self.SPAN_PAD_IDX]
            answer_end_indices = [self.SPAN_PAD_IDX]

        return answer_start_indices, answer_end_indices

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
        answer_start_indices, answer_end_indices = self._calculate_answer_indices(
            row, offset, start_idx, end_idx
        )
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


class SquadForBERTTensorizerForKD(SquadForBERTTensorizer):
    class Config(SquadForBERTTensorizer.Config):
        start_logits_column: str = "start_logits"
        end_logits_column: str = "end_logits"
        has_answer_logits_column: str = "has_answer_logits"
        pad_mask_column: str = "pad_mask"
        segment_labels_column: str = "segment_labels"

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        return super().from_config(
            config,
            start_logits_column=config.start_logits_column,
            end_logits_column=config.end_logits_column,
            has_answer_logits_column=config.has_answer_logits_column,
            pad_mask_column=config.pad_mask_column,
            segment_labels_column=config.segment_labels_column,
        )

    def __init__(
        self,
        start_logits_column=Config.start_logits_column,
        end_logits_column=Config.end_logits_column,
        has_answer_logits_column=Config.has_answer_logits_column,
        pad_mask_column=Config.pad_mask_column,
        segment_labels_column=Config.segment_labels_column,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.start_logits_column = start_logits_column
        self.end_logits_column = end_logits_column
        self.has_answer_logits_column = has_answer_logits_column
        self.pad_mask_column = pad_mask_column
        self.segment_labels_column = segment_labels_column

        # For logging
        self.total = 0
        self.mismatches = 0

    def __del__(self):
        print("Destroying SquadForBERTTensorizerForKD object")
        print(f"SquadForBERTTensorizerForKD: Number of rows read: {self.total}")
        print(f"SquadForBERTTensorizerForKD: Number of rows dropped: {self.mismatches}")

    def numberize(self, row):
        self.total += 1
        numberized_row_tuple = super().numberize(row)
        tup = numberized_row_tuple + (
            self._get_token_logits(
                row[self.start_logits_column], row[self.pad_mask_column]
            ),
            self._get_token_logits(
                row[self.end_logits_column], row[self.pad_mask_column]
            ),
            row[self.has_answer_logits_column],
        )

        try:
            assert len(tup[0]) == len(tup[6])
        except AssertionError:
            self.mismatches += 1
            print(
                f"len(tup[0]) = {len(tup[0])} and len(tup[6]) = {len(tup[6])}",
                flush=True,
            )
            raise
        return tup

    def tensorize(self, batch):
        (
            tokens,
            segment_labels,
            seq_lens,
            positions,
            answer_start_idx,
            answer_end_idx,
            start_logits,
            end_logits,
            has_answer_logits,
        ) = zip(*batch)

        tensor_tuple = super().tensorize(
            zip(
                tokens,
                segment_labels,
                seq_lens,
                positions,
                answer_start_idx,
                answer_end_idx,
            )
        )
        return tensor_tuple + (
            pad_and_tensorize(start_logits, dtype=torch.float),
            pad_and_tensorize(end_logits, dtype=torch.float),
            pad_and_tensorize(
                has_answer_logits,
                dtype=torch.float,
                pad_shape=[len(has_answer_logits), len(has_answer_logits[0])],
            ),
        )

    def _get_token_logits(self, logits, pad_mask):
        try:
            pad_start = pad_mask.index(self.vocab.get_pad_index())
        except ValueError:  # pad_index doesn't exits in pad_mask
            pad_start = len(logits)
        return logits[:pad_start]


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

        config.vocab_file = (
            resources.roberta.RESOURCE_MAP[config.vocab_file]
            if config.vocab_file in resources.roberta.RESOURCE_MAP
            else config.vocab_file
        )
        with PathManager.open(config.vocab_file) as file_path:
            vocab = build_fairseq_vocab(
                vocab_file=file_path,
                special_token_replacements={
                    "<pad>": SpecialTokens.PAD,
                    "<s>": SpecialTokens.BOS,
                    "</s>": SpecialTokens.EOS,
                    "<unk>": SpecialTokens.UNK,
                    "<mask>": SpecialTokens.MASK,
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

    def _lookup_tokens(self, text: str, seq_len: int = None):
        # BoS token is added explicitly in numberize()
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=None,
            eos_token=self.vocab.eos_token,
            max_seq_len=seq_len if seq_len else self.max_seq_len,
        )

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
