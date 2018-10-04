#!/usr/bin/env python3

from typing import Dict, List

import pandas as pd
from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.featurizer import Featurizer, InputKeys, OutputKeys
from pytext.config.component import create_featurizer
from pytext.fields import DocLabelField, Field, RawField, SeqFeatureField
from pytext.utils import data_utils
from .data_handler import DataHandler
from .joint_data_handler import JointModelDataHandler


SEQ_LENS = "seq_lens"


class SeqModelDataHandler(JointModelDataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.UTTERANCE,
        ]
        pretrained_embeds_file: str = ""

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
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
            DatasetFieldName.TEXT_FIELD: SeqFeatureField(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        }

        labels: Dict[str, Field] = {}
        if label_config.doc_label:
            labels[DatasetFieldName.DOC_LABEL_FIELD] = DocLabelField()
        extra_fields: Dict[str, Field] = {
            DatasetFieldName.TOKEN_RANGE_PAIR: RawField(),
            DatasetFieldName.INDEX_FIELD: RawField(),
            DatasetFieldName.UTTERANCE_FIELD: RawField(),
        }

        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            extra_fields=extra_fields,
            featurizer=create_featurizer(config.featurizer, feature_config),
            shuffle=config.shuffle,
            train_path=config.train_path,
            eval_path=config.eval_path,
            test_path=config.test_path,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            test_batch_size=config.test_batch_size,
        )

    def __init__(
        self, featurizer: Featurizer, **kwargs
    ) -> None:

        super().__init__(featurizer=featurizer, **kwargs)
        # configs
        self.featurizer = featurizer

        self.df_to_example_func_map = {
            # features
            DatasetFieldName.TEXT_FIELD: lambda row, field: [
                utterence.tokens for utterence in row[DFColumn.MODEL_FEATS]
            ],
            # labels
            DatasetFieldName.DOC_LABEL_FIELD: DFColumn.DOC_LABEL,
            DatasetFieldName.INDEX_FIELD: self.DF_INDEX,
            DatasetFieldName.UTTERANCE_FIELD: DFColumn.UTTERANCE,
        }

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        sequences = [
            [
                (utterence, "")
                for utterence in data_utils.parse_json_array(row[DFColumn.UTTERANCE])
            ]
            for _, row in df.iterrows()
        ]

        df[DFColumn.MODEL_FEATS] = pd.Series(
            [
                [
                    self.featurizer.featurize({
                        InputKeys.RAW_TEXT: utterence,
                        InputKeys.TOKEN_FEATURES: raw_dict,
                    })[OutputKeys.FEATURES]
                    for (utterence, raw_dict) in sequence
                ]
                for sequence in sequences
            ]
        )

        df[DFColumn.TOKEN_RANGE_PAIR] = [
            [
                data_utils.parse_token(
                    utterence, model_feat.tokenRanges
                )
                for (utterence, model_feat)
                in zip(row[DFColumn.UTTERANCE], row[DFColumn.MODEL_FEATS])
            ]
            for _, row in df.iterrows()
        ]
        return df
