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
    SeqFeatureField,
    TextFeatureField,
    WordLabelField,
)
from pytext.utils import data_utils

from .joint_data_handler import JointModelDataHandler


SEQ_LENS = "seq_lens"


class ContextualIntentSlotModelDataHandler(JointModelDataHandler):
    class Config(ConfigBase, JointModelDataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
            DFColumn.DOC_WEIGHT,
            DFColumn.WORD_WEIGHT,
        ]

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
        DatasetFieldName.DICT_FIELD,
        DatasetFieldName.CHAR_FIELD,
        DatasetFieldName.PRETRAINED_MODEL_EMBEDDING,
        DatasetFieldName.SEQ_FIELD,
    ]

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        **kwargs,
    ):
        word_feat_config = feature_config.word_feat
        features: Dict[str, Field] = {
            DatasetFieldName.SEQ_FIELD: SeqFeatureField(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            ),
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            ),
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
            labels[DatasetFieldName.DOC_LABEL_FIELD] = DocLabelField()
        if label_config.word_label:
            labels[DatasetFieldName.WORD_LABEL_FIELD] = WordLabelField(
                use_bio_labels=label_config.word_label.use_bio_labels
            )
        extra_fields: Dict[str, Field] = {
            DatasetFieldName.DOC_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.WORD_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.RAW_WORD_LABEL: RawField(),
            DatasetFieldName.TOKEN_RANGE: RawField(),
            DatasetFieldName.INDEX_FIELD: RawField(),
            DatasetFieldName.UTTERANCE_FIELD: RawField(),
        }

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
            **kwargs,
        )

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        sequence = data_utils.parse_json_array(row_data[DFColumn.UTTERANCE])

        # ignore dictionary feature for context sentences other than the last one
        features_list = [
            self.featurizer.featurize(InputRecord(raw_text=utterance))
            for utterance in sequence[:-1]
        ]

        # adding dictionary feature for the last (current) message
        features_list.append(
            self.featurizer.featurize(
                InputRecord(
                    raw_text=sequence[-1],
                    raw_gazetteer_feats=row_data.get(DFColumn.DICT_FEAT, ""),
                )
            )
        )

        res = {
            # features
            DatasetFieldName.SEQ_FIELD: [
                utterance.tokens for utterance in features_list
            ],
            DatasetFieldName.TEXT_FIELD: features_list[-1].tokens,
            DatasetFieldName.DICT_FIELD: (
                features_list[-1].gazetteer_feats,
                features_list[-1].gazetteer_feat_weights,
                features_list[-1].gazetteer_feat_lengths,
            ),
            DatasetFieldName.CHAR_FIELD: features_list[-1].characters,
            DatasetFieldName.PRETRAINED_MODEL_EMBEDDING: features_list[
                -1
            ].pretrained_token_embedding,
            # labels
            DatasetFieldName.DOC_LABEL_FIELD: row_data[DFColumn.DOC_LABEL],
            # extra data
            # TODO move the logic to FloatField
            DatasetFieldName.DOC_WEIGHT_FIELD: row_data.get(DFColumn.DOC_WEIGHT) or 1.0,
            DatasetFieldName.WORD_WEIGHT_FIELD: row_data.get(DFColumn.WORD_WEIGHT)
            or 1.0,
            DatasetFieldName.RAW_WORD_LABEL: row_data[DFColumn.WORD_LABEL],
            DatasetFieldName.INDEX_FIELD: idx,
            DatasetFieldName.UTTERANCE_FIELD: row_data[DFColumn.UTTERANCE],
            DatasetFieldName.TOKEN_RANGE: features_list[-1].token_ranges,
        }
        if DatasetFieldName.WORD_LABEL_FIELD in self.labels:
            # TODO move it into word label field
            res[DatasetFieldName.WORD_LABEL_FIELD] = data_utils.align_slot_labels(
                features_list[-1].token_ranges,
                row_data[DFColumn.WORD_LABEL],
                self.labels[DatasetFieldName.WORD_LABEL_FIELD].use_bio_labels,
            )
        return res
