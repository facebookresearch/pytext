#!/usr/bin/env python3
from typing import Dict, List, NamedTuple
from assistant.lib.feat.ttypes import ModelFeatures


class TokenFeatures(NamedTuple):
    token_index: int
    features: Dict[str, float]


class InputRecord(NamedTuple):
    raw_text: str
    token_features: List[TokenFeatures]
    locale: str = ""


class Featurizer(object):
    def featurize(self, inputRecord: InputRecord) -> ModelFeatures:
        pass

    def featurize_batch(self, inputRecords: List[InputRecord]) -> List[ModelFeatures]:
        return [self.featurize(input) for input in inputRecords]
