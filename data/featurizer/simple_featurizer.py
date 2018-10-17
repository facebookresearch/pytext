#!/usr/bin/env python3

from typing import Optional, Tuple, Sequence

from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig
from pytext.data.featurizer import Featurizer, InputRecord, OutputRecord


class SimpleFeaturizer(Featurizer):
    """
    Simple featurizer for basic tokenization and gazetteer feature alignment.
    """

    class Config(ConfigBase):
        sentence_markers: Optional[Tuple[str, str]] = None
        lowercase_tokens: bool = True

    @classmethod
    def from_config(cls, config: Config, feature_config: FeatureConfig, *kwargs):
        return cls(
            sentence_markers=config.sentence_markers,
            lowercase_tokens=feature_config.word_feat.lowercase_tokens,
        )

    def __init__(
        self, sentence_markers: Tuple[str, str] = None, lowercase_tokens: bool = True
    ) -> None:
        self.sentence_markers = sentence_markers
        self.lowercase_tokens = lowercase_tokens

    def tokenize(self, input_record: InputRecord) -> OutputRecord:
        """Tokenize one instance/example only."""
        # Dumb tokenization split on space.
        tokens = input_record.raw_text.split()
        if self.lowercase_tokens:
            tokens = list(map(str.lower, tokens))
        if self.sentence_markers:
            tokens.insert(0, self.sentence_markers[0])
            tokens.append(self.sentence_markers[1])

        # Token ranges computed based off of the dumb tokenization.
        token_ranges = []
        start, end = 0, -1
        for token in tokens:
            end = start + len(token)
            token_ranges.append(start)
            token_ranges.append(end)
            start = end + 1

        return OutputRecord(tokens=tokens, token_ranges=token_ranges)

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
