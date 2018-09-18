#!/usr/bin/env python3

import itertools
from typing import List, Tuple

import pandas as pd
import torch
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.fields import Field, TextFeatureField
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

    def __init__(
        self, featurizer: SharedFeaturizer, bptt_len: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.featurizer = featurizer
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
            featurizer=SharedFeaturizer(),
            bptt_len=bptt_len,
            raw_columns=columns,
            features=features,
            labels={},
            extra_fields={},
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
        # NOTE that currently featurizer will lower case all tokens
        df[DFColumn.MODEL_FEATS] = pd.Series(
            self.featurizer.featurize_parallel(
                df[DFColumn.RAW_FEATS].tolist(), self.num_workers
            )
        )

        def featurize(df):
            return list(
                itertools.chain.from_iterable(
                    row.tokens for row in df[DFColumn.MODEL_FEATS]
                )
            )

        ret_df = pd.DataFrame()
        ret_df[DFColumn.UTTERANCE] = pd.Series([featurize(df)])
        return ret_df

    def _get_train_batch(
        self, datasets: Tuple[textdata.Dataset, ...], batch_size: Tuple[int, ...]
    ) -> Tuple[BatchIterator, ...]:
        return tuple(
            BatchIterator(iter, self._postprocess_batch)
            for iter in textdata.BPTTIterator.splits(
                datasets,
                batch_sizes=batch_size,
                bptt_len=self.bptt_len,
                device="cuda:0" if cuda_utils.CUDA_ENABLED else "cpu",
                sort_within_batch=True,
                repeat=False,
                sort_key=self.sort_key,
            )
        )

    def _input_from_batch(self, batch):
        return (
            # (bsx x seq_len)
            batch.text.t().contiguous(),
            torch.Tensor([batch.text.size(0)] * batch.text.size(1)).type_as(batch.text),
        )

    def _target_from_batch(self, batch):
        return (batch.target.t().contiguous(),)

    def get_test_batch(self, file_path: str, batch_size: int) -> BatchIterator:
        test_data = self.gen_dataset_from_path(file_path)
        return BatchIterator(
            textdata.BPTTIterator(
                test_data,
                batch_size=batch_size,
                bptt_len=self.bptt_len,
                device="cuda:0" if cuda_utils.CUDA_ENABLED else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
        )
