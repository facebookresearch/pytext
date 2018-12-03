#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.config.field_config import DocLabelConfig
from pytext.config.pair_classification import (
    ExtraField,
    ModelInput,
    ModelInputConfig,
    TargetConfig,
)
from pytext.data.featurizer import InputRecord
from pytext.fields import (
    DocLabelField,
    Field,
    RawField,
    TextFeatureField,
    create_fields,
    create_label_fields,
)

from .data_handler import DataHandler


class RawData:
    DOC_LABEL = "doc_label"
    TEXT1 = "text1"
    TEXT2 = "text2"


class PairClassificationDataHandler(DataHandler):
    class Config(DataHandler.Config):
        columns_to_read: List[str] = [RawData.DOC_LABEL, RawData.TEXT1, RawData.TEXT2]

    def sort_key(self, example) -> Any:
        return len(getattr(example, ModelInput.TEXT1))

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: ModelInputConfig,
        target_config: TargetConfig,
        **kwargs,
    ):
        features: Dict[str, Field] = create_fields(
            feature_config,
            {ModelInput.TEXT1: TextFeatureField, ModelInput.TEXT2: TextFeatureField},
        )
        assert len(features) == 2
        # share the processing field
        features[ModelInput.TEXT2] = features[ModelInput.TEXT1]

        labels: Dict[str, Field] = create_label_fields(
            target_config, {DocLabelConfig._name: DocLabelField}
        )
        extra_fields: Dict[str, Field] = {ExtraField.UTTERANCE_PAIR: RawField()}
        kwargs.update(config.items())
        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            extra_fields=extra_fields,
            **kwargs,
        )

    def _train_input_from_batch(self, batch):
        # token1, token2, seq_len1, seq_len2
        return batch.text1[0], batch.text2[0], batch.text1[1], batch.text2[1]

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            ModelInput.TEXT1: self.featurizer.featurize(
                InputRecord(raw_text=row_data[RawData.TEXT1])
            ).tokens,
            ModelInput.TEXT2: self.featurizer.featurize(
                InputRecord(raw_text=row_data[RawData.TEXT2])
            ).tokens,
            DocLabelConfig._name: row_data[RawData.DOC_LABEL],
            ExtraField.UTTERANCE_PAIR: f"{row_data[RawData.TEXT1]} | {row_data[RawData.TEXT2]}",
        }
