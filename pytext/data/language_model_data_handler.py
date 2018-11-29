#!/usr/bin/env python3

from typing import Any, Dict, List

import torch
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config.field_config import FeatureConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import Field, RawField, TextFeatureField

from .data_handler import DataHandler


FEATURE_ITOS_MAP = "feature_itos_map"


class LanguageModelDataHandler(DataHandler):
    class Config(DataHandler.Config):
        columns_to_read: List[str] = [DFColumn.UTTERANCE]
        append_bos: bool = True
        append_eos: bool = True

    @classmethod
    def from_config(
        cls, config: Config, feature_config: FeatureConfig, *args, **kwargs
    ):
        # For language modeling the only input is a collection of utterances.
        # The input and the labels are created by the LangaugeModelDataHandler.
        # The input at time step t+1 becomes a label for the input at time step t.
        word_feat_config = feature_config.word_feat
        features: Dict[str, Field] = {
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                eos_token=VocabMeta.EOS_TOKEN if config.append_eos else None,
                init_token=VocabMeta.INIT_TOKEN if config.append_bos else None,
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        }
        labels: Dict[str, Field] = {}
        extra_fields: Dict[str, Field] = {DatasetFieldName.UTTERANCE_FIELD: RawField()}
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
        raw_input = InputRecord(raw_text=row_data[DFColumn.UTTERANCE])

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
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        return (
            text_input[0][:, 0:-1].contiguous(),
            torch.min(text_input[1], text_input[1].max() - 1),
        )

    def _target_from_batch(self, batch):
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        return text_input[0][:, 1:].contiguous()

    def _context_from_batch(self, batch):
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        # batch.text[1] is the length of each sequence
        res = {
            DatasetFieldName.SEQ_LENS: torch.min(
                text_input[1], text_input[1].max() - 1
            ),
            DatasetFieldName.TARGET_SEQ_LENS: text_input[1] - 1,
        }
        res.update(super()._context_from_batch(batch))
        return res
