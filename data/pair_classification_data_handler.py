#!/usr/bin/env python3

from typing import Dict, List, Any

from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.fields import DocLabelField, Field, RawField, TextFeatureField
from pytext.utils import data_utils

from .data_handler import DataHandler


TEXT_2 = "text_2"
UTTERANCE_PAIR = "utterance"


class PairClassificationDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DatasetFieldName.TEXT_FIELD,
            TEXT_2,
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
            TEXT_2: text_field,
        }
        extra_fields: Dict[str, Field] = {DatasetFieldName.UTTERANCE_FIELD: RawField()}

        labels: Dict[str, Field] = {}
        if label_config.doc_label:
            labels[DatasetFieldName.DOC_LABEL_FIELD] = DocLabelField()

        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            shuffle=config.shuffle,
            extra_fields=extra_fields,
            train_path=config.train_path,
            eval_path=config.eval_path,
            test_path=config.test_path,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            test_batch_size=config.test_batch_size,
            **kwargs,
        )

    def _train_input_from_batch(self, batch):
        return tuple(zip(*(getattr(batch, name) for name in self.features)))

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        row_data[
            UTTERANCE_PAIR
        ] = f"{row_data[DatasetFieldName.TEXT_FIELD]} | {row_data[TEXT_2]}"
        return row_data
