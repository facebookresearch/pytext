#!/usr/bin/env python3

from typing import List

import pandas as pd
import torch
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.fields import Field, RawField, TextFeatureField
from pytext.utils import data_utils

from .data_handler import DataHandler


FEATURE_ITOS_MAP = "feature_itos_map"


class LanguageModelDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [DFColumn.UTTERANCE]

    def __init__(
        self, featurizer: SharedFeaturizer, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.featurizer = featurizer

        self.df_to_feat_func_map = {
            # features
            DatasetFieldName.TEXT_FIELD: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].tokens
        }

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        **kwargs
    ):
        # For language modeling the only input is a collection of utterances.
        # The input and the labels are created by the LangaugeModelDataHandler.
        # The input at time step t+1 becomes a label for the input at time step t.
        word_feat_config = feature_config.word_feat
        features: List[Field] = [
            TextFeatureField(
                DatasetFieldName.TEXT_FIELD,
                eos_token=VocabMeta.EOS_TOKEN,
                init_token=VocabMeta.INIT_TOKEN,
                export_names=word_feat_config.export_input_names,
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        ]
        labels: List[Field] = []
        extra_fields: List[Field] = [RawField(DatasetFieldName.TOKEN_RANGE_PAIR)]
        return cls(
            featurizer=SharedFeaturizer(),
            raw_columns=config.columns_to_read,
            features=features,
            labels=labels,
            extra_fields=extra_fields,
        )

    def _gen_extra_metadata(self):
        # a bit hacky here, the label vocab is just the word token vocab
        self.metadata.labels = {
            "label": self.metadata.features[DatasetFieldName.TEXT_FIELD]
        }

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if DFColumn.DICT_FEAT not in df:
            df[DFColumn.DICT_FEAT] = ""
        df[DFColumn.RAW_FEATS] = df.apply(
            lambda row: (row[DFColumn.UTTERANCE], row[DFColumn.DICT_FEAT]), axis=1
        )

        df[DFColumn.MODEL_FEATS] = pd.Series(
            self.featurizer.featurize_parallel(
                df[DFColumn.RAW_FEATS].tolist(), self.num_workers
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
        # batch.text[1] is the length of each sequence
        # length of the longest sequences will be subtracted by 1, but for other
        # smaller sequences, it will remain the same
        # Example Batch:
        # [[how, are, you],
        #  [hello, world, <pad>]]
        # Input for the above batch will be:
        # [[how, are],
        #  [hello, world]]
        return (
            batch.text[0][:, 0:-1].contiguous(),
            torch.min(batch.text[1], batch.text[1].max() - 1),
        )

    def _target_from_batch(self, batch):
        return batch.text[0][:, 1:].contiguous()

    def _context_from_batch(self, batch):
        # batch.text[1] is the length of each sequence
        res = {SEQ_LENS: batch.text[1]}
        res.update(super()._context_from_batch(batch))
        return res
