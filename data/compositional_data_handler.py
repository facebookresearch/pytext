#!/usr/bin/env python3
from typing import Dict, List

import pandas as pd
from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import ActionField, DictFeatureField, Field, TextFeatureField
from pytext.utils import data_utils

from .data_handler import DataHandler


TREE_COLUMN = "tree"
ACTION_FEATURE_FIELD = "action_idx_feature"
ACTION_LABEL_FIELD = "action_idx_label"


class CompositionalDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
        DatasetFieldName.DICT_FIELD,
        ACTION_FEATURE_FIELD,
    ]

    @classmethod
    def from_config(cls, config: Config, feature_config: FeatureConfig, **kwargs):
        word_feat_config = feature_config.word_feat
        features: Dict[str, Field] = {
            # TODO assuming replacing numbers with NUM and unkify be done in featurizer
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        }
        if feature_config.dict_feat and feature_config.dict_feat.embed_dim > 0:
            features[DatasetFieldName.DICT_FIELD] = DictFeatureField()
        features[ACTION_FEATURE_FIELD] = ActionField()
        return cls(
            raw_columns=config.columns_to_read,
            features=features,
            labels={ACTION_LABEL_FIELD: ActionField()},
            train_path=config.train_path,
            eval_path=config.eval_path,
            test_path=config.test_path,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            test_batch_size=config.test_batch_size,
            **kwargs
        )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.df_to_example_func_map = {
            # TODO set_tokens_indices, should implement another field
            # TODO is it the same with the original tokens seq?
            DatasetFieldName.TEXT_FIELD: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].tokens,
            DatasetFieldName.DICT_FIELD: lambda row, field: (
                row[DFColumn.MODEL_FEATS].gazetteer_feats,
                row[DFColumn.MODEL_FEATS].gazetteer_feat_weights,
                row[DFColumn.MODEL_FEATS].gazetteer_feat_lengths,
            ),
            ACTION_FEATURE_FIELD: lambda row, field: row[TREE_COLUMN].to_actions(),
            ACTION_LABEL_FIELD: lambda row, field: row[TREE_COLUMN].to_actions(),
        }

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if DFColumn.DICT_FEAT not in df:
            df[DFColumn.DICT_FEAT] = ""

        df[DFColumn.RAW_FEATS] = df.apply(
            lambda row: InputRecord(
                raw_text=row[DFColumn.UTTERANCE],
                raw_gazetteer_feats=row[DFColumn.DICT_FEAT],
            ),
            axis=1,
        )

        df[DFColumn.MODEL_FEATS] = pd.Series(
            output
            for output in self.featurizer.featurize_batch(
                df[DFColumn.RAW_FEATS].tolist()
            )
        )

        df[DFColumn.TOKEN_RANGE_PAIR] = [
            data_utils.parse_token(
                row[DFColumn.UTTERANCE], row[DFColumn.MODEL_FEATS].token_ranges
            )
            for _, row in df.iterrows()
        ]

        return df

    def _input_from_batch(self, batch, is_train):
        # text_input[0] is contains numericalized tokens.
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        return (text_input[0], text_input[1]) + tuple(
            getattr(batch, name, None)
            for name in self.FULL_FEATURES
            if name != DatasetFieldName.TEXT_FIELD
        )
