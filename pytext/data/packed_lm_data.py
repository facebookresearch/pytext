#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Type

from pytext.common.constants import Stage
from pytext.data import Batcher, Data
from pytext.data.bert_tensorizer import BERTTensorizerBase
from pytext.data.data import RowData
from pytext.data.sources import DataSource
from pytext.data.tensorizers import Tensorizer, TokenTensorizer


class PackedLMData(Data):
    """
    Special purpose Data object which assumes a single text tensorizer.  Packs
    tokens into a square batch with no padding.  Used for LM training. The object
    also takes in an optional language argument which is used for cross-lingual
    LM training.
    """

    __EXPANSIBLE__ = True

    class Config(Data.Config):
        max_seq_len: int = 128

    @classmethod
    def from_config(
        cls,
        config: Config,
        schema: Dict[str, Type],
        tensorizers: Dict[str, Tensorizer],
        language: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        init_tensorizers: Optional[bool] = True,
    ):
        return super(PackedLMData, cls).from_config(
            config,
            schema,
            tensorizers,
            rank,
            world_size,
            language=language,
            max_seq_len=config.max_seq_len,
            init_tensorizers=init_tensorizers,
        )

    def __init__(
        self,
        data_source: DataSource,
        tensorizers: Dict[str, Tensorizer],
        batcher: Batcher = None,
        max_seq_len: int = Config.max_seq_len,
        sort_key: Optional[str] = None,
        # language is used in cross-lingual LM training
        language: Optional[str] = None,
        in_memory: Optional[bool] = False,
        init_tensorizers: Optional[bool] = True,
    ):
        super().__init__(
            data_source, tensorizers, batcher, sort_key, in_memory, init_tensorizers
        )
        assert len(list(self.tensorizers.items())) == 1
        self.tensorizer_name, self.tensorizer = list(self.tensorizers.items())[0]
        self.remainder: Dict[str, List[int]] = {"tokens": [], "segment_labels": []}
        self.max_seq_len = max_seq_len
        self.language = language
        self.batch = {Stage.TRAIN: None, Stage.EVAL: None, Stage.TEST: None}

    def _parse_row(self, row):
        """
        The output of numberization has different number of elements depending on
        the tensorizer used. For example: positions tensor is only output by the
        XLMTensorizer. This function unpacks the elements according to the
        specific tensorizer used.
        Additionally, since we are packing tokens into fixed size
        blocks, we don't need to use the positions vector output by the call to
        numberize. We will simply create this in `_format_output_row`.
        """
        numberized_row = self.tensorizer.numberize(row)
        if isinstance(self.tensorizer, BERTTensorizerBase):
            tokens, segment_labels, seq_len, _ = numberized_row
        elif isinstance(self.tensorizer, TokenTensorizer):
            tokens, seq_len, _ = numberized_row
            segment_labels = []
        else:
            raise NotImplementedError(
                "PackedLMData only supports XLMTensorizer, BERTTensorizer and "
                "TokenTensorizer."
            )
        return tokens, segment_labels, seq_len

    def _format_output_row(self, tokens, segment_labels, seq_len):
        """
        The tensorize function for different tensorizers takes in different
        number of inputs which may be arranged differently. This function formats
        the output dict to conform to the expectations of the tensorizer.
        In case of the XLMTensorizer, we also need to create a new positions list
        which goes from 0 to seq_len.
        """
        if isinstance(self.tensorizer, BERTTensorizerBase):
            positions = [index for index in range(seq_len)]
            return {self.tensorizer_name: (tokens, segment_labels, seq_len, positions)}
        elif isinstance(self.tensorizer, TokenTensorizer):
            # dummy token_ranges
            return {self.tensorizer_name: (tokens, seq_len, [(-1, -1)] * seq_len)}
        else:
            raise NotImplementedError(
                "PackedLMData only supports BERTTensorizer and TokenTensorizer."
            )

    def _yield_and_reset(self, row):
        packed_tokens = list(self.remainder["tokens"])
        packed_segments = list(self.remainder["segment_labels"])
        self.remainder: Dict[str, List[int]] = {"tokens": [], "segment_labels": []}
        return RowData(
            row,
            self._format_output_row(packed_tokens, packed_segments, len(packed_tokens)),
        )

    def numberize_rows(self, rows):
        last_row = None
        """
        This function does the actual packing. It processes rows until we obtain
        a block of data with length = max_seq_len.
        """
        for row in rows:
            last_row = row

            # if the packedLM object has a language member then a cross-lingual
            # LM is being trained using monolingual data.
            # Add this language to the row since the underlying
            # tensorizer needs this to generate language embeddings (used as
            # segment_labels below)
            if self.language:
                row["language"] = self.language

            tokens, segment_labels, seq_len = self._parse_row(row)
            remaining = self.max_seq_len - len(self.remainder["tokens"]) - 1
            while remaining < len(tokens):
                self.remainder["tokens"].extend(tokens[:remaining])
                self.remainder["segment_labels"].extend(segment_labels[:remaining])
                tokens = tokens[remaining:]
                segment_labels = segment_labels[remaining:]
                # packed LM data doesn't respect data cardinality,
                # therefore, it stores the row at the start position,
                # instead of the exact corresponding row.
                yield self._yield_and_reset(row)
                remaining = self.max_seq_len - 1
            self.remainder["tokens"].extend(tokens)
            self.remainder["segment_labels"].extend(segment_labels)
        if len(self.remainder["tokens"]):
            yield self._yield_and_reset(last_row)
