#!/usr/bin/env python3
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from joblib import Parallel, delayed
from pytext.config.component import Component, ComponentType


class InputRecord(NamedTuple):
    """Input data contract between Featurizer and DataHandler."""
    raw_text: str
    raw_gazetteer_feats: str = ""
    locale: str = ""


class OutputRecord(NamedTuple):
    """Output data contract between Featurizer and DataHandler."""
    tokens: List[str]
    token_ranges: Optional[List[int]] = None
    gazetteer_feats: Optional[List[str]] = None
    gazetteer_feat_lengths: Optional[List[int]] = None
    gazetteer_feat_weights: Optional[List[float]] = None
    characters: Optional[List[List[str]]] = None
    pretrained_token_embedding: Optional[List[List[float]]] = None


class Featurizer(Component):
    """
    Featurizer is tasked with performing data preprocessing that should be shared
    between training and inference, namely, tokenization and gaztteer features
    alignment.

    This is an interface whose featurize() method must be implemented so that
    the implemented interface can be used with the appropriate data handler.
    """

    __COMPONENT_TYPE__ = ComponentType.FEATURIZER

    def featurize(self, input_record: InputRecord) -> OutputRecord:
        raise NotImplementedError("Featurizer.featurize() method must be implemented.")

    def featurize_batch(
        self, input_record_list: Sequence[InputRecord]
    ) -> Sequence[OutputRecord]:
        """Featurize a batch of instances/examples.
        """
        return [self.featurize(record) for record in input_record_list]
