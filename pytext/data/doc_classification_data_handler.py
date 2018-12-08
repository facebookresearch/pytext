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
    FloatVectorField,
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
    """
    The `DocClassificationDataHandler` prepares the data for document
    classififcation. Each sentence is read line by line with its label as the
    target.
    """

    class Config(DataHandler.Config):
        """
        Configuration class for `DocClassificationDataHandler`.

        Attributes:
            columns_to_read (List[str]): List containing the names of the
                columns to read from the data files.
            max_seq_len (int): Maximum sequence length for the input. The input
                is trimmed after the maximum sequence length.
        """

        columns_to_read: List[str] = cls_vars(RawData)
        max_seq_len: int = -1

    @classmethod
    def from_config(
        cls,
        config: Config,
        model_input_config: ModelInputConfig,
        target_config: TargetConfig,
        **kwargs,
    ):
        """
        Factory method to construct an instance of `DocClassificationDataHandler`
        from the module's config object and feature config object.

        Args:
            config (DocClassificationDataHandler.Config): Configuration object
                specifying all the parameters of `DocClassificationDataHandler`.
            model_input_config (ModelInputConfig): Configuration object
                specifying all the parameters of the model config.
            target_config (TargetConfig): Configuration object specifying all
                the parameters of the target.

        Returns:
            type: An instance of `DocClassificationDataHandler`.
        """
        model_input_fields: Dict[str, Field] = create_fields(
            model_input_config,
            {
                ModelInput.WORD_FEAT: TextFeatureField,
                ModelInput.DICT_FEAT: DictFeatureField,
                ModelInput.CHAR_FEAT: CharFeatureField,
                ModelInput.PRETRAINED_MODEL_EMBEDDING: PretrainedModelEmbeddingField,
                ModelInput.DENSE_FEAT: FloatVectorField,
            },
        )
        target_fields: Dict[str, Field] = create_label_fields(
            target_config, {DocLabelConfig._name: DocLabelField}
        )
        extra_fields: Dict[str, Field] = {ExtraField.RAW_TEXT: RawField()}
        kwargs.update(config.items())
        return cls(
            raw_columns=config.columns_to_read,
            labels=target_fields,
            features=model_input_fields,
            extra_fields=extra_fields,
            **kwargs,
        )

    def _get_tokens(self, mode_feature):
        if self.max_seq_len > 0:
            # truncate tokens if max_seq_len is set
            return mode_feature.tokens[: self.max_seq_len]
        else:
            return mode_feature.tokens

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.featurizer.featurize(
            InputRecord(
                raw_text=row_data.get(RawData.TEXT, ""),
                raw_gazetteer_feats=row_data.get(RawData.DICT_FEAT, ""),
            )
        )
        res = {
            # feature
            ModelInput.WORD_FEAT: self._get_tokens(features),
            ModelInput.DICT_FEAT: (
                features.gazetteer_feats,
                features.gazetteer_feat_weights,
                features.gazetteer_feat_lengths,
            ),
            ModelInput.CHAR_FEAT: features.characters,
            ModelInput.PRETRAINED_MODEL_EMBEDDING: features.pretrained_token_embedding,
            ModelInput.DENSE_FEAT: row_data.get(ModelInput.DENSE_FEAT),
            # target
            DocLabelConfig._name: row_data.get(RawData.DOC_LABEL),
            # extra data
            ExtraField.RAW_TEXT: row_data.get(RawData.TEXT),
        }
        return res

    def _train_input_from_batch(self, batch):
        word_feat_input = getattr(batch, ModelInput.WORD_FEAT)
        result = [
            word_feat_input[0],  # token indices
            *(
                getattr(batch, name, None)
                for name in self.features
                if name not in {ModelInput.WORD_FEAT, ModelInput.DENSE_FEAT}
            ),
            word_feat_input[1],  # seq lens
        ]
        # Append any inputs to decoder layer at the end. (Only 1 right now)
        if ModelInput.DENSE_FEAT in self.features:
            result.append(getattr(batch, ModelInput.DENSE_FEAT))
        return tuple(result)
