#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, NamedTuple, Optional, Sequence, Tuple

from pytext.config.component import Component, ComponentType
from pytext.config.field_config import FeatureConfig


class InputRecord(NamedTuple):
    """Input data contract between Featurizer and DataHandler."""

    raw_text: str
    raw_gazetteer_feats: str = ""
    locale: str = ""


class OutputRecord(NamedTuple):
    """Output data contract between Featurizer and DataHandler."""

    tokens: List[str]
    token_ranges: Optional[List[Tuple[int]]] = None
    gazetteer_feats: Optional[List[str]] = None
    gazetteer_feat_lengths: Optional[List[int]] = None
    gazetteer_feat_weights: Optional[List[float]] = None
    characters: Optional[List[List[str]]] = None
    contextual_token_embedding: Optional[List[List[float]]] = None
    dense_feats: Optional[List[float]] = None


class Featurizer(Component):
    """
    Featurizer is tasked with performing data preprocessing that should be shared
    between training and inference, namely, tokenization and gazetteer features
    alignment.

    This is an interface whose featurize() method must be implemented so that
    the implemented interface can be used with the appropriate data handler.
    """

    __COMPONENT_TYPE__ = ComponentType.FEATURIZER
    __EXPANSIBLE__ = True

    @classmethod
    def from_config(cls, config, feature_config: FeatureConfig):
        return cls(config, feature_config)

    def __init__(self, config, feature_config: FeatureConfig) -> None:
        super().__init__(config)
        self.feature_config = feature_config

    def featurize(self, input_record: InputRecord) -> OutputRecord:
        raise NotImplementedError("Featurizer.featurize() method must be implemented.")

    def featurize_batch(
        self, input_record_list: Sequence[InputRecord]
    ) -> Sequence[OutputRecord]:
        """Featurize a batch of instances/examples.
        """
        return [self.featurize(record) for record in input_record_list]

    def get_sentence_markers(self, locale=None):
        raise NotImplementedError(
            "Featurizer.get_sentence_markers() method must be implemented."
        )
