#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.config.doc_classification import (
    ExtraField,
    ModelInput,
    ModelInputConfig,
    TargetConfig,
)
from pytext.config.field_config import DocLabelConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import (
    CharFeatureField,
    DictFeatureField,
    DocLabelField,
    Field,
    PretrainedModelEmbeddingField,
    RawField,
    TextFeatureField,
    create_fields,
    create_label_fields,
)
from pytext.utils.python_utils import cls_vars

from .data_handler import DataHandler


class RawData:
    DOC_LABEL = "doc_label"
    TEXT = "text"
    DICT_FEAT = "dict_feat"


class DocClassificationDataHandler(DataHandler):
    class Config(DataHandler.Config):
        columns_to_read: List[str] = cls_vars(RawData)

    @classmethod
    def from_config(
        cls,
        config: Config,
        model_input_config: ModelInputConfig,
        target_config: TargetConfig,
        **kwargs,
    ):
        model_input_fields: Dict[str, Field] = create_fields(
            model_input_config,
            {
                ModelInput.WORD_FEAT: TextFeatureField,
                ModelInput.DICT_FEAT: DictFeatureField,
                ModelInput.CHAR_FEAT: CharFeatureField,
                ModelInput.PRETRAINED_MODEL_EMBEDDING: PretrainedModelEmbeddingField,
            },
        )
        target_fields: Dict[str, Field] = create_label_fields(
            target_config, {DocLabelConfig._name: DocLabelField}
        )
        extra_fields: Dict[str, Field] = {
            ExtraField.INDEX: RawField(),
            ExtraField.RAW_TEXT: RawField(),
        }
        kwargs.update(config.items())
        return cls(
            raw_columns=config.columns_to_read,
            labels=target_fields,
            features=model_input_fields,
            extra_fields=extra_fields,
            **kwargs,
        )

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        features = self.featurizer.featurize(
            InputRecord(
                raw_text=row_data.get(RawData.TEXT, ""),
                raw_gazetteer_feats=row_data.get(RawData.DICT_FEAT, ""),
            )
        )
        res = {
            # feature
            ModelInput.WORD_FEAT: features.tokens,
            ModelInput.DICT_FEAT: (
                features.gazetteer_feats,
                features.gazetteer_feat_weights,
                features.gazetteer_feat_lengths,
            ),
            ModelInput.CHAR_FEAT: features.characters,
            ModelInput.PRETRAINED_MODEL_EMBEDDING: features.pretrained_token_embedding,
            # target
            DocLabelConfig._name: row_data.get(RawData.DOC_LABEL),
            # extra data
            ExtraField.INDEX: idx,
            ExtraField.RAW_TEXT: row_data.get(RawData.TEXT),
        }
        return res

    def _train_input_from_batch(self, batch):
        word_feat_input = getattr(batch, ModelInput.WORD_FEAT)
        return (
            word_feat_input[0],  # token indices
            *(
                getattr(batch, name, None)
                for name in self.features
                if name != ModelInput.WORD_FEAT
            ),
            word_feat_input[1],  # seq lens
        )
