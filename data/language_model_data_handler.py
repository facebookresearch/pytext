#!/usr/bin/env python3

from typing import Dict, List, Any

import torch
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import Field, RawField, TextFeatureField

from .data_handler import DataHandler


FEATURE_ITOS_MAP = "feature_itos_map"


class LanguageModelDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [DFColumn.UTTERANCE]

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
        features: Dict[str, Field] = {
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                eos_token=VocabMeta.EOS_TOKEN,
                init_token=VocabMeta.INIT_TOKEN,
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        }
        labels: Dict[str, Field] = {}
        extra_fields: Dict[str, Field] = {
            DatasetFieldName.UTTERANCE_FIELD: RawField(),
        }
        return cls(
            raw_columns=config.columns_to_read,
            features=features,
            labels=labels,
            extra_fields=extra_fields,
            train_path=config.train_path,
            eval_path=config.eval_path,
            test_path=config.test_path,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            test_batch_size=config.test_batch_size,
            **kwargs
        )

    def _gen_extra_metadata(self):
        # a bit hacky here, the label vocab is just the word token vocab
        self.metadata.labels = {
            "label": self.metadata.features[DatasetFieldName.TEXT_FIELD]
        }

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        raw_input = InputRecord(
            raw_text=row_data[DFColumn.UTTERANCE]
        )

        features = self.featurizer.featurize(raw_input)

        return {
            # features
            DatasetFieldName.TEXT_FIELD: features.tokens,
            DatasetFieldName.UTTERANCE_FIELD: row_data[DFColumn.UTTERANCE],
        }

    def _train_input_from_batch(self, batch):
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
        res = {
            DatasetFieldName.SEQ_LENS: torch.min(
                batch.text[1], batch.text[1].max() - 1
            ),
            DatasetFieldName.TARGET_SEQ_LENS: batch.text[1] - 1,
        }
        res.update(super()._context_from_batch(batch))
        return res
