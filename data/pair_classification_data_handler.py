#!/usr/bin/env python3

from typing import Dict, List

import pandas as pd
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config import ConfigBase
from pytext.config.field_config import LabelConfig
from pytext.fields import DocLabelField, Field, TextFeatureField
from pytext.models.embeddings.shared_token_embedding import SharedTokenEmbedding
from pytext.utils import data_utils

from .data_handler import DataHandler


SEQ_LENS = "seq_lens"


class PairClassificationDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [DFColumn.DOC_LABEL, DFColumn.UTTERANCE, "text_2"]

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: SharedTokenEmbedding.Config,
        label_config: LabelConfig,
        **kwargs,
    ):
        word_feat_config = feature_config.word_feat
        text_field = TextFeatureField(
            eos_token=VocabMeta.EOS_TOKEN,
            init_token=VocabMeta.INIT_TOKEN,
            tokenize=data_utils.simple_tokenize,
            pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
            embed_dim=word_feat_config.embed_dim,
            embedding_init_strategy=word_feat_config.embedding_init_strategy,
            vocab_file=word_feat_config.vocab_file,
            vocab_size=word_feat_config.vocab_size,
            vocab_from_train_data=word_feat_config.vocab_from_train_data,
        )
        features: Dict[str, Field] = {
            DatasetFieldName.TEXT_FIELD: text_field,
            "text_2": text_field,
        }

        labels: Dict[str, Field] = {}
        if label_config.doc_label:
            labels[DatasetFieldName.DOC_LABEL_FIELD] = DocLabelField()

        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            shuffle=config.shuffle,
        )

    def _input_from_batch(self, batch):
        return tuple(zip(*(getattr(batch, name) for name in self.features)))

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _context_from_batch(self, batch):
        # add together the lengths of the two sequences
        res = {SEQ_LENS: batch.text[1] + batch.text_2[1]}
        res.update(super()._context_from_batch(batch))
        return res
