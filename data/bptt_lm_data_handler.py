#!/usr/bin/env python3

import itertools
import math
from typing import Any, Dict, List

import torch
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import TextFeatureField
from pytext.utils import cuda_utils
from torchtext import data as textdata

from .data_handler import BatchIterator, DataHandler


FEATURE_ITOS_MAP = "feature_itos_map"


class BPTTLanguageModelDataHandler(DataHandler):
    """ BPTTLanguageModelDataHandler treats data as a single document, concatenating
    all tokens together. BPTTIterator arranges the dataset into columns of batch size
    and subdivides the source data into chunks of length bptt_len. It enables
    hidden state of ith batch carried over to (i+1)th batch.
    """

    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [DFColumn.UTTERANCE]
        bptt_len: int = 35

    def __init__(self, bptt_len: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bptt_len = bptt_len

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
        columns = config.columns_to_read
        bptt_len = config.bptt_len
        if bptt_len <= 0:
            raise TypeError("BPTT Sequence length cannot be 0 or less.")
        features = {
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                eos_token=VocabMeta.EOS_TOKEN, include_lengths=False
            )
        }
        return cls(
            bptt_len=bptt_len,
            raw_columns=columns,
            features=features,
            labels={},
            extra_fields={},
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

    def preprocess(self, data: List[Dict[str, Any]]):
        return [
            {
                DFColumn.UTTERANCE: list(
                    itertools.chain.from_iterable(super().preprocess(data))
                )
            }
        ]

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> List[str]:
        return self.featurizer.featurize(
            InputRecord(raw_text=row_data[DFColumn.UTTERANCE])
        ).tokens

    def _get_train_iter(
        self,
        train_dataset: textdata.Dataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> BatchIterator:
        dataset_shard = self._get_dataset_shard(train_dataset, rank, world_size)
        num_all_batches = math.ceil(len(train_dataset) / float(batch_size))
        return BatchIterator(
            textdata.BPTTIterator(
                dataset_shard,
                batch_size=batch_size,
                bptt_len=self.bptt_len,
                device="cuda:{}".format(torch.cuda.curren_device())
                if cuda_utils.CUDA_ENABLED
                else "cpu",
                sort_within_batch=True,
                repeat=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
            num_batches=math.ceil(num_all_batches / float(world_size)),
        )

    def _train_input_from_batch(self, batch):
        return (
            # (bsx x seq_len)
            batch.text.t().contiguous(),
            torch.Tensor([batch.text.size(0)] * batch.text.size(1)).type_as(batch.text),
        )

    def _target_from_batch(self, batch):
        return (batch.target.t().contiguous(),)

    def get_test_iter(self, file_path: str, batch_size: int) -> BatchIterator:
        test_data = self.gen_dataset_from_path(file_path)
        return BatchIterator(
            textdata.BPTTIterator(
                test_data,
                batch_size=batch_size,
                bptt_len=self.bptt_len,
                device="cuda:{}".format(torch.cuda.curren_device())
                if cuda_utils.CUDA_ENABLED
                else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
        )
