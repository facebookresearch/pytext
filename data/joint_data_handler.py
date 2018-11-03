#!/usr/bin/env python3

from typing import Any, Dict, List

from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
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
)
from pytext.utils import data_utils

from .data_handler import DataHandler


SEQ_LENS = "seq_lens"


class JointModelDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
            DFColumn.DOC_WEIGHT,
            DFColumn.WORD_WEIGHT,
        ]
        max_seq_len: int = -1

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
        DatasetFieldName.DICT_FIELD,
        DatasetFieldName.CHAR_FIELD,
        DatasetFieldName.PRETRAINED_MODEL_EMBEDDING,
    ]

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        **kwargs
    ):
        word_feat_config = feature_config.word_feat
        features: Dict[str, Field] = {
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        }
        if feature_config.dict_feat:
            features[DatasetFieldName.DICT_FIELD] = DictFeatureField()

        if feature_config.char_feat:
            features[DatasetFieldName.CHAR_FIELD] = CharFeatureField()

        labels: Dict[str, Field] = {}
        if feature_config.pretrained_model_embedding:
            features[
                DatasetFieldName.PRETRAINED_MODEL_EMBEDDING
            ] = PretrainedModelEmbeddingField()

        if label_config.doc_label:
            labels[DatasetFieldName.DOC_LABEL_FIELD] = DocLabelField(
                getattr(label_config.doc_label, "label_weights", None)
            )
        if label_config.word_label:
            labels[DatasetFieldName.WORD_LABEL_FIELD] = WordLabelField(
                use_bio_labels=label_config.word_label.use_bio_labels
            )
        extra_fields: Dict[str, Field] = {
            DatasetFieldName.DOC_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.WORD_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.TOKEN_RANGE: RawField(),
            DatasetFieldName.INDEX_FIELD: RawField(),
            DatasetFieldName.UTTERANCE_FIELD: RawField(),
        }
        if label_config.word_label:
            extra_fields[DatasetFieldName.RAW_WORD_LABEL] = RawField()

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
            max_seq_len=config.max_seq_len,
            **kwargs
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

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
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
            DatasetFieldName.INDEX_FIELD: idx,
            DatasetFieldName.UTTERANCE_FIELD: row_data.get(DFColumn.UTTERANCE),
            DatasetFieldName.TOKEN_RANGE: features.token_ranges,
        }
        if DatasetFieldName.DOC_LABEL_FIELD in self.labels:
            res[DatasetFieldName.DOC_LABEL_FIELD] = row_data[DFColumn.DOC_LABEL]
        if DatasetFieldName.WORD_LABEL_FIELD in self.labels:
            # TODO move it into word label field
            res[DatasetFieldName.WORD_LABEL_FIELD] = data_utils.align_slot_labels(
                features.token_ranges,
                row_data[DFColumn.WORD_LABEL],
                self.labels[DatasetFieldName.WORD_LABEL_FIELD].use_bio_labels,
            )
            res[DatasetFieldName.RAW_WORD_LABEL] = row_data[DFColumn.WORD_LABEL]
        return res

    def _train_input_from_batch(self, batch):
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        # text_input[1] is the length of each word
        return (text_input[0], ) + tuple(
            getattr(batch, name, None)
            for name in self.features
            if name != DatasetFieldName.TEXT_FIELD
        ) + (text_input[1],)

    def _context_from_batch(self, batch):
        # text_input[1] is the length of each word
        res = {SEQ_LENS: getattr(batch, DatasetFieldName.TEXT_FIELD)[1]}
        res.update(super()._context_from_batch(batch))
        return res

    def _gen_extra_metadata(self):
        self.metadata.tokenizer_config_dict = getattr(
            self.featurizer, "tokenizer_config_dict", {}
        )
