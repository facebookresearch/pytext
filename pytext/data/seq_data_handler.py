#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config import ConfigBase
from pytext.config.field_config import DocLabelConfig, FeatureConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import DocLabelField, Field, RawField, SeqFeatureField
from pytext.utils import data_utils

from .joint_data_handler import JointModelDataHandler


SEQ_LENS = "seq_lens"


class SeqModelDataHandler(JointModelDataHandler):
    class Config(JointModelDataHandler.Config):
        columns_to_read: List[str] = [DFColumn.DOC_LABEL, DFColumn.UTTERANCE]
        pretrained_embeds_file: str = ""

    FULL_FEATURES = [DatasetFieldName.TEXT_FIELD]

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: FeatureConfig,
        label_config: DocLabelConfig,
        **kwargs
    ):
        word_feat_config = feature_config.word_feat
        features: Dict[str, Field] = {
            DatasetFieldName.TEXT_FIELD: SeqFeatureField(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        }
        labels: Dict[str, Field] = {DocLabelConfig._name: DocLabelField()}
        extra_fields: Dict[str, Field] = {DatasetFieldName.UTTERANCE_FIELD: RawField()}

        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            extra_fields=extra_fields,
            shuffle=config.shuffle,
            train_path=config.train_path,
            eval_path=config.eval_path,
            test_path=config.test_path,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            test_batch_size=config.test_batch_size,
            **kwargs
        )

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        sequence = data_utils.parse_json_array(row_data[DFColumn.UTTERANCE])

        features_list = [
            self.featurizer.featurize(InputRecord(raw_text=utterance))
            for utterance in sequence
        ]

        return {
            # features
            DatasetFieldName.TEXT_FIELD: [
                utterance.tokens for utterance in features_list
            ],
            # labels
            DatasetFieldName.DOC_LABEL_FIELD: row_data[DFColumn.DOC_LABEL],
            DatasetFieldName.UTTERANCE_FIELD: row_data[DFColumn.UTTERANCE],
        }
