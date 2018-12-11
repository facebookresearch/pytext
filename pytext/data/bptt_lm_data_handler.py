#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import math
from typing import Any, Dict, List

import torch
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config.field_config import FeatureConfig, WordLabelConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import TextFeatureField
from pytext.utils import cuda_utils
from torchtext import data as textdata

from .data_handler import BatchIterator, DataHandler


class BPTTLanguageModelDataHandler(DataHandler):
    """
    `BPTTLanguageModelDataHandler` treats data as a single document, concatenating
    all tokens together. BPTTIterator arranges the dataset into columns of
    batch size and subdivides the source data into chunks of length bptt_len.
    It enables hidden state of ith batch carried over to (i+1)th batch.

    Args:
        bptt_len (int) : Input sequence length to backpropagate to.
    """

    class Config(DataHandler.Config):
        """
        Configuration class for `BPTTLanguageModelDataHandler`.

        Attributes:
            columns_to_read (List[str]): List containing the names of the
                columns to read from the data files.
            bptt_len (int): Input sequence length to backpropagate to.
        """

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
        label_config: WordLabelConfig,
        **kwargs
    ):
        """
        Factory method to construct an instance of `BPTTLanguageModelDataHandler`
        from the module's config object and feature config object.

        Args:
            config (LanguageModelDataHandler.Config): Configuration object
                specifying all the parameters of `BPTTLanguageModelDataHandler`.
            feature_config (FeatureConfig): Configuration object specifying all
                the parameters of all input features.

        Returns:
            type: An instance of `BPTTLanguageModelDataHandler`.
        """
        # For language modeling the only input is a collection of utterances.
        # The input and the labels are created by the LangaugeModelDataHandler.
        # The input at time step t+1 becomes a label for the input at time step t.
        columns = config.columns_to_read
        bptt_len = config.bptt_len
        if bptt_len <= 0:
            raise TypeError("BPTT Sequence length cannot be 0 or less.")
        features = {
            # the name must be text because it's hardcoded in torchtext BPTT iterator
            "text": TextFeatureField(
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
            pass_index=False,
            **kwargs
        )

    def init_feature_metadata(
        self,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ):
        """
        Prepares the metadata for the language model features.
        """
        super().init_feature_metadata(train_data, eval_data, test_data)
        # workaround the hardcoded torchtext name
        self.metadata.features[DatasetFieldName.TEXT_FIELD] = self.metadata.features[
            "text"
        ]

    def init_target_metadata(
        self,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ):
        """
        Prepares the metadata for the language model target.
        """
        self.metadata.target = self.metadata.features[DatasetFieldName.TEXT_FIELD]

    def preprocess(self, data: List[Dict[str, Any]]):
        tokens = []
        for row in data:
            tokens.extend(self.preprocess_row(row))
        return [{"text": tokens}]

    def preprocess_row(self, row_data: Dict[str, Any]) -> List[str]:
        """
        Preprocess steps for a single input row.

        Args:
            row_data (Dict[str, Any]): Dict representing the input row and
                columns.

        Returns:
            List[str]: List of tokens.
        """
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
        dataset_shard, max_num_examples = self._get_dataset_shard(
            train_dataset, rank, world_size
        )
        # Compute the per-worker batch size
        assert (
            batch_size >= world_size
        ), "batch size needs to be >= the distributed world size"
        batch_size = batch_size // world_size

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
            num_batches=math.ceil(max_num_examples / float(batch_size)),
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
        """
        Get test data iterator from test data file.

        Args:
            file_path (str): Path to test data file.
            batch_size (int): Batch size

        Returns:
            BatchIterator: An instance of BatchIterator to iterate over the
                supplied test data file.
        """
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
