#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from typing import List, Optional, Sequence, Tuple

from pytext.common.constants import VocabMeta
from pytext.config import ConfigBase
from pytext.data.featurizer import Featurizer, InputRecord, OutputRecord


class SimpleFeaturizer(Featurizer):
    """
    Simple featurizer for basic tokenization and gazetteer feature alignment.
    """

    class Config(ConfigBase):
        sentence_markers: Optional[Tuple[str, str]] = None
        lowercase_tokens: bool = True
        split_regex: str = r"\s+"
        convert_to_bytes: bool = False

    def tokenize(self, input_record: InputRecord) -> OutputRecord:
        """Tokenize one instance/example only."""
        tokens: List[str] = []
        token_ranges: List[Tuple[int, int]] = []

        def add_token(text, start, end):
            token = text[start:end]
            if token:
                tokens.append(token)
                token_ranges.append((start, end))

        if self.config.convert_to_bytes:
            start = 0
            text = input_record.raw_text.encode()
            for byte in text:
                tokens.append(chr(byte))
                token_ranges.append((start, start + 1))
                start += 1
        else:
            start = 0
            text = input_record.raw_text
            for sep in re.finditer(self.config.split_regex, text):
                add_token(text, start, sep.start())
                start = sep.end()
            add_token(text, start, len(text))

        if not tokens:
            # Add PAD_TOKEN in case of empty text
            tokens = [VocabMeta.PAD_TOKEN]
        if self.config.lowercase_tokens:
            tokens = list(map(str.lower, tokens))
        if self.config.sentence_markers:
            tokens.insert(0, self.config.sentence_markers[0])
            tokens.append(self.config.sentence_markers[1])

        characters = [list(tok) for tok in tokens]

        # TODO: support remaining features (see OutputRecord)
        return OutputRecord(
            tokens=tokens, token_ranges=token_ranges, characters=characters
        )

    def tokenize_batch(
        self, input_records: Sequence[InputRecord]
    ) -> Sequence[OutputRecord]:
        return [self.tokenize(in_record) for in_record in input_records]

    def featurize(self, input_record: InputRecord) -> OutputRecord:
        """Featurize one instance/example only."""
        return self.tokenize(input_record)

    def featurize_batch(
        self, input_records: Sequence[InputRecord]
    ) -> Sequence[OutputRecord]:
        return [self.featurize(in_record) for in_record in input_records]

    def get_sentence_markers(self, locale=None):
        return self.config.sentence_markers
