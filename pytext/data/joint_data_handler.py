#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Union

from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config.field_config import (
    DocLabelConfig,
    FeatureConfig,
    TargetConfigBase,
    WordLabelConfig,
)
from pytext.data.featurizer import InputRecord
from pytext.fields import (
    CharFeatureField,
    DictFeatureField,
    DocLabelField,
    Field,
    FloatField,
    PretrainedModelEmbeddingField,
    RawField,
    TextFeatureField,
    WordLabelField,
    create_fields,
    create_label_fields,
)
from pytext.utils import data_utils

from .data_handler import DataHandler


SEQ_LENS = "seq_lens"


class JointModelDataHandler(DataHandler):
    class Config(DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
            DFColumn.DOC_WEIGHT,
            DFColumn.WORD_WEIGHT,
        ]
        max_seq_len: int = -1

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: FeatureConfig,
        label_configs: Union[DocLabelConfig, WordLabelConfig, List[TargetConfigBase]],
        **kwargs,
    ):
        features: Dict[str, Field] = create_fields(
            feature_config,
            {
                DatasetFieldName.TEXT_FIELD: TextFeatureField,
                DatasetFieldName.DICT_FIELD: DictFeatureField,
                DatasetFieldName.CHAR_FIELD: CharFeatureField,
                DatasetFieldName.PRETRAINED_MODEL_EMBEDDING: PretrainedModelEmbeddingField(),
            },
        )

        # Label fields.
        labels: Dict[str, Field] = create_label_fields(
            label_configs,
            {
                DocLabelConfig._name: DocLabelField,
                WordLabelConfig._name: WordLabelField,
            },
        )
        has_word_label = WordLabelConfig._name in labels

        extra_fields: Dict[str, Field] = {
            DatasetFieldName.DOC_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.WORD_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.TOKEN_RANGE: RawField(),
            DatasetFieldName.UTTERANCE_FIELD: RawField(),
        }
        if has_word_label:
            extra_fields[DatasetFieldName.RAW_WORD_LABEL] = RawField()

        kwargs.update(config.items())
        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            extra_fields=extra_fields,
            **kwargs,
        )

    def _get_tokens(self, mode_feature):
        if self.max_seq_len > 0:
            # truncate tokens if max_seq_len is set
            return mode_feature.tokens[: self.max_seq_len]
        else:
            return mode_feature.tokens

    def featurize(self, row_data: Dict[str, Any]):
        return self.featurizer.featurize(
            InputRecord(
                raw_text=row_data.get(DFColumn.UTTERANCE, ""),
                raw_gazetteer_feats=row_data.get(DFColumn.DICT_FEAT, ""),
            )
        )

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.featurize(row_data)
        res = {
            # feature field
            # TODO move the logic to text field
            DatasetFieldName.TEXT_FIELD: self._get_tokens(features),
            DatasetFieldName.DICT_FIELD: (
                features.gazetteer_feats,
                features.gazetteer_feat_weights,
                features.gazetteer_feat_lengths,
            ),
            DatasetFieldName.CHAR_FIELD: features.characters,
            DatasetFieldName.PRETRAINED_MODEL_EMBEDDING: features.pretrained_token_embedding,
            # extra data
            # TODO move the logic to FloatField
            DatasetFieldName.DOC_WEIGHT_FIELD: row_data.get(DFColumn.DOC_WEIGHT) or 1.0,
            DatasetFieldName.WORD_WEIGHT_FIELD: row_data.get(DFColumn.WORD_WEIGHT)
            or 1.0,
            DatasetFieldName.UTTERANCE_FIELD: row_data.get(DFColumn.UTTERANCE),
            DatasetFieldName.TOKEN_RANGE: features.token_ranges,
        }
        if DatasetFieldName.DOC_LABEL_FIELD in self.labels:
            res[DatasetFieldName.DOC_LABEL_FIELD] = row_data.get(DFColumn.DOC_LABEL)
        if DatasetFieldName.WORD_LABEL_FIELD in self.labels:
            # TODO move it into word label field
            res[DatasetFieldName.WORD_LABEL_FIELD] = data_utils.align_slot_labels(
                features.token_ranges,
                row_data.get(DFColumn.WORD_LABEL),
                self.labels[DatasetFieldName.WORD_LABEL_FIELD].use_bio_labels,
            )
            res[DatasetFieldName.RAW_WORD_LABEL] = row_data.get(DFColumn.WORD_LABEL)
        return res

    def _train_input_from_batch(self, batch):
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        # text_input[1] is the length of each word
        return (
            (text_input[0],)
            + tuple(
                getattr(batch, name, None)
                for name in self.features
                if name != DatasetFieldName.TEXT_FIELD
            )
            + (text_input[1],)
        )

    def _context_from_batch(self, batch):
        # text_input[1] is the length of each word
        res = {SEQ_LENS: getattr(batch, DatasetFieldName.TEXT_FIELD)[1]}
        res.update(super()._context_from_batch(batch))
        return res

    def _gen_extra_metadata(self):
        self.metadata.tokenizer_config_dict = getattr(
            self.featurizer, "tokenizer_config_dict", {}
        )
