#!/usr/bin/env python3

from typing import Dict, List

import pandas as pd
from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.featurizer import Featurizer
from pytext.fb.data.assistant_featurizer import (
    AssistantFeaturizer,
    parse_assistant_raw_record,
)
from pytext.fields import (
    CapFeatureField,
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

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
        DatasetFieldName.DICT_FIELD,
        DatasetFieldName.CAP_FIELD,
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

        if feature_config.cap_feat:
            features[DatasetFieldName.CAP_FIELD] = CapFeatureField()
        if feature_config.char_feat:
            features[DatasetFieldName.CHAR_FIELD] = CharFeatureField()

        labels: Dict[str, Field] = {}
        model_file_paths = None
        if feature_config.pretrained_model_embedding:
            features[
                DatasetFieldName.PRETRAINED_MODEL_EMBEDDING
            ] = PretrainedModelEmbeddingField()
            model_file_paths = (
                feature_config.pretrained_model_embedding.model_file_paths
            )

        if label_config.doc_label:
            labels[DatasetFieldName.DOC_LABEL_FIELD] = DocLabelField(
                getattr(label_config.doc_label, 'label_weights', None)
            )
        if label_config.word_label:
            labels[DatasetFieldName.WORD_LABEL_FIELD] = WordLabelField(
                use_bio_labels=label_config.word_label.use_bio_labels
            )
        extra_fields: Dict[str, Field] = {
            DatasetFieldName.DOC_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.WORD_WEIGHT_FIELD: FloatField(),
            DatasetFieldName.RAW_WORD_LABEL: RawField(),
            DatasetFieldName.TOKEN_RANGE_PAIR: RawField(),
            DatasetFieldName.INDEX_FIELD: RawField(),
            DatasetFieldName.UTTERANCE_FIELD: RawField(),
        }

        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            extra_fields=extra_fields,
            featurizer=AssistantFeaturizer(pre_trained_models_dict=model_file_paths),
            shuffle=config.shuffle,
        )

    def __init__(self, featurizer: Featurizer, **kwargs) -> None:

        super().__init__(**kwargs)
        # configs
        self.featurizer = featurizer

        self.df_to_example_func_map = {
            # features
            DatasetFieldName.TEXT_FIELD: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].tokens,
            DatasetFieldName.DICT_FIELD: lambda row, field: (
                row[DFColumn.MODEL_FEATS].dictFeats,
                row[DFColumn.MODEL_FEATS].dictFeatWeights,
                row[DFColumn.MODEL_FEATS].dictFeatLengths,
            ),
            DatasetFieldName.CHAR_FIELD: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].chars,
            DatasetFieldName.PRETRAINED_MODEL_EMBEDDING: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].pretrainedTokenEmbedding,
            DatasetFieldName.CAP_FIELD: lambda row, field: [
                data_utils.capitalization_feature(t)
                for (t, (_, __)) in row[DFColumn.TOKEN_RANGE_PAIR]
            ],
            # labels
            DatasetFieldName.DOC_LABEL_FIELD: DFColumn.DOC_LABEL,
            DatasetFieldName.WORD_LABEL_FIELD: lambda row, field: data_utils.align_slot_labels(
                row[DFColumn.TOKEN_RANGE_PAIR],
                row[DFColumn.WORD_LABEL],
                field.use_bio_labels,
            ),
            # extra context
            DatasetFieldName.DOC_WEIGHT_FIELD: lambda row, field: row.get(
                DFColumn.DOC_WEIGHT
            )
            or 1.0,
            DatasetFieldName.WORD_WEIGHT_FIELD: lambda row, field: row.get(
                DFColumn.WORD_WEIGHT
            )
            or 1.0,
            DatasetFieldName.RAW_WORD_LABEL: DFColumn.WORD_LABEL,
            DatasetFieldName.INDEX_FIELD: self.DF_INDEX,
            DatasetFieldName.UTTERANCE_FIELD: DFColumn.UTTERANCE,
        }

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if DFColumn.DICT_FEAT not in df:
            df[DFColumn.DICT_FEAT] = ""

        df[DFColumn.RAW_FEATS] = df.apply(
            lambda row: parse_assistant_raw_record(
                row[DFColumn.UTTERANCE],
                row[DFColumn.DICT_FEAT]
            ),
            axis=1
        )

        df[DFColumn.MODEL_FEATS] = pd.Series(
            output["features_obj"]
            for output in self.featurizer.featurize_batch(
                df[DFColumn.RAW_FEATS].tolist()
            )
        )

        df[DFColumn.TOKEN_RANGE_PAIR] = [
            data_utils.parse_token(
                row[DFColumn.UTTERANCE], row[DFColumn.MODEL_FEATS].tokenRanges
            )
            for _, row in df.iterrows()
        ]

        return df

    def _input_from_batch(self, batch):
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        # text_input[1] is the length of each word
        return (text_input[0], text_input[1]) + tuple(
            getattr(batch, name, None)
            for name in self.FULL_FEATURES
            if name != DatasetFieldName.TEXT_FIELD
        )

    def _context_from_batch(self, batch):
        # text_input[1] is the length of each word
        res = {SEQ_LENS: getattr(batch, DatasetFieldName.TEXT_FIELD)[1]}
        res.update(super()._context_from_batch(batch))
        return res

    def _gen_extra_metadata(self):
        self.metadata.tokenizer_config_path_dict = (
            self.featurizer.tokenizer_config_path_dict
        )
