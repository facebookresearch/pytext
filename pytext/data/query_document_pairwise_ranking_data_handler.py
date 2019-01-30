#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.config.query_document_pairwise_ranking import ModelInput, ModelInputConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import Field, TextFeatureField

from .data_handler import DataHandler


class QueryDocumentPairwiseRankingDataHandler(DataHandler):
    class Config(DataHandler.Config):
        columns_to_read: List[str] = [
            ModelInput.QUERY,
            ModelInput.POS_RESPONSE,
            ModelInput.NEG_RESPONSE,
        ]

    def sort_key(self, example) -> Any:
        return len(getattr(example, ModelInput.POS_RESPONSE))

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: ModelInputConfig,
        target_config: None,
        **kwargs,
    ):
        features: Dict[str, Field] = {
            ModelInput.POS_RESPONSE: TextFeatureField.from_config(
                feature_config.pos_response
            )
        }
        # we want vocab to be built once across all fields
        # so we just make all features point to POS_RESPONSE
        # TODO: if we're reading pretrained embeddings, they
        # will be read multiple times
        features[ModelInput.NEG_RESPONSE] = features[ModelInput.POS_RESPONSE]
        features[ModelInput.QUERY] = features[ModelInput.POS_RESPONSE]
        assert len(features) == 3, "Expected three text features"

        kwargs.update(config.items())
        return cls(
            raw_columns=config.columns_to_read, labels=None, features=features, **kwargs
        )

    def _train_input_from_batch(self, batch):
        # token1, token2, seq_len1, seq_len2
        return (
            batch.pos_response[0],
            batch.neg_response[0],
            batch.query[0],
            batch.pos_response[1],
            batch.neg_response[1],
            batch.query[1],
        )

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            ModelInput.POS_RESPONSE: self.featurizer.featurize(
                InputRecord(raw_text=row_data[ModelInput.POS_RESPONSE])
            ).tokens,
            ModelInput.NEG_RESPONSE: self.featurizer.featurize(
                InputRecord(raw_text=row_data[ModelInput.NEG_RESPONSE])
            ).tokens,
            ModelInput.QUERY: self.featurizer.featurize(
                InputRecord(raw_text=row_data[ModelInput.QUERY])
            ).tokens,
        }
