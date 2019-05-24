#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import math
import multiprocessing
import os
from copy import deepcopy
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from pytext.common.constants import BatchContext, DatasetFieldName, DFColumn, VocabMeta
from pytext.config.component import Component, ComponentType
from pytext.config.field_config import Target
from pytext.config.pytext_config import ConfigBase
from pytext.data.featurizer import Featurizer
from pytext.fields import Field, FieldMeta, RawField, VocabUsingField
from pytext.utils import cuda, distributed, embeddings as embeddings_utils
from pytext.utils.data import parse_json_array
from torchtext import data as textdata

from .utils import align_target_labels


class CommonMetadata:
    features: Dict[str, FieldMeta]
    target: FieldMeta
    dataset_sizes: Dict[str, int]


class BatchIterator:
    """
    BatchIterator is a wrapper of TorchText. Iterator that provide flexibility to
    map batched data to a tuple of (input, target, context) and other additional
    steps such as dealing with distributed training.

    Args:
        batches (Iterator[TorchText.Batch]): iterator of TorchText.Batch, which
            shuffles/batches the data in __iter__ and return a batch of data in
            __next__
        processor: function to run after getting batched data from TorchText.Iterator,
            the function should define a way to map to data into
            (input, target, context)
        include_input (bool): if input data should be returned, default is true
        include_target (bool): if target data should be returned, default is true
        include_context (bool): if context data should be returned, default is true
        is_train (bool): if the batch data is for training
        num_batches (int): total batches to generate, this param if for distributed
            training due to a limitation in PyTorch's distributed training backend
            that enforces all the parallel workers to have the same number of batches
            we workaround it by adding dummy batches at the end
    """

    def __init__(
        self,
        batches,
        processor,
        include_input=True,
        include_target=True,
        include_context=True,
        is_train=True,
        num_batches=0,
    ):
        self.processor = processor
        self.batches = batches
        self.include_input = include_input
        self.include_target = include_target
        self.include_context = include_context
        self.is_train = is_train
        self.total_num_batches = num_batches

    def __iter__(self):
        """
        Iterate Torchtext.Iterator, map batch data into (input, target, context)
        tuple and generate dummy batches for distributed training

        Returns:
            input: tuple of tensors that can be fed directly into model forward
                function
            target: tensor or tuple of tensors as the model target for computing
                loss
            context: any extra info to be used in downstream steps, can be any
                data type
        """
        num_batches = len(self.batches)
        for i, batch in enumerate(self.batches):
            input, target, context = self.processor(
                batch,
                self.include_input,
                self.include_target,
                self.include_context,
                self.is_train,
            )
            yield (input, target, context)
            # Due to a limitation in PyTorch's distributed training backend that
            # enforces that all the parallel workers to have the same number of
            # batches we keep yielding the last batch until the requested total
            # number of batches is fullfilled
            if i == num_batches - 1:
                context = deepcopy(context)
                context.update({BatchContext.IGNORE_LOSS: True})
                for _j in range(num_batches, len(self)):
                    yield (input, target, context)

    def __len__(self):
        return self.total_num_batches


class DataHandler(Component):
    """
    DataHandler is the central place to prepare data for model training/testing.
    The class is responsible of:

    * Define pipeline to process data and generate batch of tensors to be
      consumed by model. Each batch is a (input, target, extra_data) tuple, in
      which input can be feed directly into model.
    * Initialize global context, such as build vocab, load pretrained embeddings.
      Store the context as metadata, and provide function to serialize/deserialize
      the metadata

    The data processing pipeline contains the following steps:

    * Read data from file into a list of raw data examples
    * Convert each row of row data to a TorchText Example. This logic happens
      in process_row function and will:

      * Invoke featurizer, which contains data processing steps to apply
        for both training and inference time, e.g: tokenization
      * Use the raw data and results from featurizer to do any preprocessing

    * Generate a TorchText.Dataset that contains the list of Example, the Dataset
      also has a list of TorchText.Field, which defines how to do padding and
      numericalization while batching data.
    * Return a BatchIterator which will give a tuple of (input, target, context)
      tensors for each iteration. By default the tensors have a 1:1 mapping to
      the TorchText.Field fields, but this behavior can be overwritten by
      _input_from_batch, _target_from_batch, _context_from_batch functions.

    Attributes:
        raw_columns (List[str]): columns to read from data source. The order should
            match the data stored in that file.
        featurizer (Featurizer): perform data preprocessing that should be shared
            between training and inference
        features (Dict[str, Field]): a dict of name -> field that used to process data
            as model input
        labels (Dict[str, Field]): a dict of name -> field that used to process data
            as training target
        extra_fields (Dict[str, Field]): fields that process any extra data used
            neither as model input nor target. This is None by default
        text_feature_name (str): name of the text field, used to define the default
            sort key of data
        shuffle (bool): if the dataset should be shuffled, true by default
        sort_within_batch (bool): if data within same batch should be sorted, true
            by default
        train_path (str): path of training data file
        eval_path (str): path of evaluation data file
        test_path (str): path of test data file
        train_batch_size (int): training batch size, 128 by default
        eval_batch_size (int): evaluation batch size, 128 by default
        test_batch_size (int): test batch size, 128 by default
        max_seq_len (int): maximum length of tokens to keep in sequence
        pass_index (bool): if the original index of data in the batch should be
            passed along to downstream steps, default is true
    """

    class Config(ConfigBase):
        columns_to_read: List[str] = []
        shuffle: bool = True
        sort_within_batch: bool = True
        train_path: str = "train.tsv"
        eval_path: str = "eval.tsv"
        test_path: str = "test.tsv"
        train_batch_size: int = 128
        eval_batch_size: int = 128
        test_batch_size: int = 128
        column_mapping: Dict[str, str] = {}

    __COMPONENT_TYPE__ = ComponentType.DATA_HANDLER

    def __init__(
        self,
        raw_columns: List[str],
        labels: Dict[str, Field],
        features: Dict[str, Field],
        featurizer: Featurizer,
        extra_fields: Dict[str, Field] = None,
        text_feature_name: str = DatasetFieldName.TEXT_FIELD,
        shuffle: bool = True,
        sort_within_batch: bool = True,
        train_path: str = "train.tsv",
        eval_path: str = "eval.tsv",
        test_path: str = "test.tsv",
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        test_batch_size: int = 128,
        max_seq_len: int = -1,
        pass_index: bool = True,
        column_mapping: Dict[str, str] = None,
        **kwargs,
    ) -> None:
        self.raw_columns: List[str] = raw_columns or []
        self.labels: Dict[str, Field] = labels or {}
        self.features: Dict[str, Field] = features or {}
        self.featurizer = featurizer
        self.extra_fields: Dict[str, Field] = extra_fields or {}
        if pass_index:
            self.extra_fields[BatchContext.INDEX] = RawField()
        self.text_feature_name: str = text_feature_name

        self.metadata_cls: Type = CommonMetadata
        self.metadata: CommonMetadata = CommonMetadata()
        self._data_cache: MutableMapping[str, Any] = {}
        self.shuffle = shuffle
        self.sort_within_batch = sort_within_batch
        self.num_workers = multiprocessing.cpu_count()
        self.max_seq_len = max_seq_len

        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.column_mapping = column_mapping

    def load_vocab(self, vocab_file, vocab_size, lowercase_tokens: bool = False):
        """
        Loads items into a set from a file containing one item per line.
        Items are added to the set from top of the file to bottom.
        So, the items in the file should be ordered by a preference (if any), e.g.,
        it makes sense to order tokens in descending order of frequency in corpus.

        Args:
            vocab_file (str): vocab file to load
            vocab_size (int): maximum tokens to load, will only load the first n if
                the actual vocab size is larger than this parameter
            lowercase_tokens (bool): if the tokens should be lowercased
        """
        vocab: Set[str] = set()
        if os.path.isfile(vocab_file):
            with open(vocab_file, "r") as f:
                for i, line in enumerate(f):
                    if vocab_size > 0 and len(vocab) == vocab_size:
                        print(
                            f"Read {i+1} items from {vocab_file} "
                            f"to load vocab of size {vocab_size}. "
                            f"Skipping rest of the file"
                        )
                        break
                    line = line.strip()
                    vocab.add(line.lower() if lowercase_tokens else line)
        elif not vocab_file:
            print(f"{vocab_file} doesn't exist. Cannot load vocabulary from it")
        return vocab

    def sort_key(self, example: textdata.Example) -> Any:
        """
        How to sort data in every batch, default behavior is by the length of input
        text
        Args:
            example (Example): one torchtext example
        """
        return len(getattr(example, self.text_feature_name))

    def metadata_to_save(self):
        """
        Save metadata, pretrained_embeds_weight should be excluded
        """
        # make a copy
        metadata = deepcopy(self.metadata)
        # pretrained_embeds_weight takes a lot space and is not needed in inference time
        for feature_meta in metadata.features.values():
            feature_meta.pretrained_embeds_weight = None
        if isinstance(metadata.target, list):
            for target_meta in metadata.target:
                target_meta.pretrained_embeds_weight = None
        else:
            metadata.target.pretrained_embeds_weight = None
        return metadata

    def load_metadata(self, metadata: CommonMetadata):
        """
        Load previously saved metadata
        """
        self.metadata = metadata
        for name, field in self.features.items():
            if field.use_vocab and name in metadata.features:
                field.load_meta(metadata.features[name])

        target_meta = metadata.target
        if not isinstance(metadata.target, list):
            target_meta = [target_meta]
        for field, meta in zip(self.labels.values(), target_meta):
            field.load_meta(meta)

    def gen_dataset_from_path(
        self,
        path: str,
        rank: int = 0,
        world_size: int = 1,
        include_label_fields: bool = True,
        use_cache: bool = True,
    ) -> textdata.Dataset:
        """
        Generate a dataset from file
        Returns:
            dataset (TorchText.Dataset)
        """
        if use_cache and path in self._data_cache and rank == 0 and world_size == 1:
            return self._data_cache[path]

        shard_range = (
            distributed.get_shard_range(
                self.metadata.dataset_sizes[path], rank, world_size
            )
            if world_size > 1
            else None
        )
        res = self.gen_dataset(
            self.read_from_file(path, self.raw_columns),
            include_label_fields,
            shard_range,
        )
        if rank == 0 and world_size == 1:
            self._data_cache[path] = res
        return res

    def gen_dataset(
        self,
        data: Iterable[Dict[str, Any]],
        include_label_fields: bool = True,
        shard_range: Tuple[int, int] = None,
    ) -> textdata.Dataset:
        """
        Generate torchtext Dataset from raw in memory data.
        Returns:
            dataset (TorchText.Dataset)
        """
        to_process = {}
        to_process.update(self.features)
        to_process.update(self.extra_fields)
        if include_label_fields:
            to_process.update(self.labels)
        else:
            to_process.pop(Target.TARGET_LABEL_FIELD, None)
        fields = {name: (name, field) for name, field in to_process.items()}
        # generate example from dataframe
        examples = [
            textdata.Example.fromdict(row, fields)
            for idx, row in enumerate(self.preprocess(data))
            if not shard_range or shard_range[0] <= idx <= shard_range[1]
        ]
        return textdata.Dataset(examples, to_process)

    def preprocess(self, data: Iterable[Dict[str, Any]]):
        """
        preprocess the raw data to create TorchText.Example, this is the second
        step in whole processing pipeline
        Returns:
            data (Generator[Dict[str, Any]])
        """
        for idx, row in enumerate(data):
            preprocessed_row = self.preprocess_row(row)
            if preprocessed_row:
                preprocessed_row[BatchContext.INDEX] = idx
                yield preprocessed_row

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        preprocess steps for a single input row, sub class should override it
        """
        return row_data

    def init_metadata_from_path(self, train_path, eval_path, test_path):
        """
        Initialize metadata using data from file
        """
        # get data sets
        pathes = [train_path, eval_path, test_path]
        datasets = [self.gen_dataset_from_path(path) for path in pathes]
        self._init_metadata(*datasets)
        self.metadata.dataset_sizes = {
            path: len(dataset) for (path, dataset) in zip(pathes, datasets)
        }

    def init_metadata(self):
        """
        Initialize metadata using data from configured path
        """
        self.init_metadata_from_path(self.train_path, self.eval_path, self.test_path)

    def init_metadata_from_raw_data(self, *data):
        """
        Initialize metadata using in memory data
        """
        self._init_metadata(*[self.gen_dataset(d) for d in data])

    def _init_metadata(
        self,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ):
        self.init_feature_metadata(train_data, eval_data, test_data)
        self.init_target_metadata(train_data, eval_data, test_data)
        self._gen_extra_metadata()

    def init_feature_metadata(
        self,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ):
        # field metadata
        self.metadata.features = {}
        # build vocabs for features
        for name, feat in self.features.items():
            weights = None
            if feat.use_vocab:
                pretrained_embeddings = None
                pretrained_embeddings_path = getattr(
                    feat, "pretrained_embeddings_path", None
                )
                if pretrained_embeddings_path:
                    print(
                        "load pretrained embeddings from {}".format(
                            pretrained_embeddings_path
                        )
                    )
                    pretrained_embeddings = embeddings_utils.PretrainedEmbedding(
                        pretrained_embeddings_path, feat.lower
                    )

                if hasattr(feat, "vocab"):  # Don't rebuild vocab
                    print(f"Vocab for feature {name} has been built. Not rebuilding.")
                else:
                    print(f"Building vocab for feature {name}.")
                    vocab_data = self._get_data_to_build_vocab(
                        feat, train_data, eval_data, test_data, pretrained_embeddings
                    )
                    feat.build_vocab(*vocab_data, min_freq=feat.min_freq)
                print("{} field's vocabulary size is {}".format(name, len(feat.vocab)))

                # Initialize pretrained embedding weights.
                if pretrained_embeddings:
                    weights = pretrained_embeddings.initialize_embeddings_weights(
                        feat.vocab.stoi,
                        VocabMeta.UNK_TOKEN,
                        feat.embed_dim,
                        feat.embedding_init_strategy,
                    )  # this is of type torch.Tensor

            meta = feat.get_meta()
            meta.pretrained_embeds_weight = weights
            self.metadata.features[name] = meta

    def init_target_metadata(
        self,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ):
        self.metadata.target = []
        # build vocabs for label fields
        for name, label in self.labels.items():
            if name in [Target.TARGET_PROB_FIELD, Target.TARGET_LOGITS_FIELD]:
                continue
            # Need test data to make sure we cover all of the labels in it
            # It is particularly important when BIO is enabled as a B-[Label] can
            # appear in train and eval but test can have B-[Label] and I-[Label]
            weights = None
            if label.use_vocab:
                if not hasattr(label, "vocab"):  # Don't rebuild vocab
                    print("Building vocab for label {}".format(name))
                    label.build_vocab(train_data, eval_data, test_data)
                else:
                    print(f"Vocab for label {name} has been built. Not rebuilding.")
                print(
                    "{} field's vocabulary size is {}".format(
                        name, len(label.vocab.itos)
                    )
                )
                pretrained_embeddings = None
                pretrained_embeddings_path = getattr(
                    label, "pretrained_embeddings_path", None
                )
                if pretrained_embeddings_path:
                    pretrained_embeddings = embeddings_utils.PretrainedEmbedding(
                        pretrained_embeddings_path
                    )
                if pretrained_embeddings:
                    weights = pretrained_embeddings.initialize_embeddings_weights(
                        label.vocab.stoi,
                        label.unk_token,
                        label.embed_dim,
                        label.embedding_init_strategy,
                    )  # this is of type torch.Tensor

            meta = label.get_meta()
            meta.pretrained_embeds_weight = weights
            self.metadata.target.append(meta)
        if len(self.metadata.target) == 1:
            [self.metadata.target] = self.metadata.target

    def _get_data_to_build_vocab(
        self,
        feat: Field,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
        pretrained_embeddings: embeddings_utils.PretrainedEmbedding,
    ) -> List[Any]:
        """
        This method prepares the list of data sources that Field.build_vocab()
        accepts to build vocab from. Based on the specifications from `feat`, the
        data can come from

        - train data
        - eval + test data
        - specified vocab file
        - pretrained embeddings dictionary
        """
        data = []

        if isinstance(feat, VocabUsingField):
            if feat.vocab_from_all_data:
                print("Adding tokens from train, eval, and test data to vocab.")
                data.extend([train_data, eval_data, test_data])
            elif feat.vocab_from_train_data:
                print("Adding tokens from train data to vocab.")
                data.append(train_data)

        if hasattr(feat, "vocab_file") and feat.vocab_file:
            print(f"Adding tokens from {feat.vocab_file} to vocab.")
            lowercase_tokens = feat.lower if hasattr(feat, "lower") else False
            assert feat.min_freq == 1
            vocab_set = self.load_vocab(
                feat.vocab_file, feat.vocab_size, lowercase_tokens
            )
            if vocab_set:
                data.append([vocab_set])

        if getattr(feat, "vocab_from_pretrained_embeddings", False):
            print("Adding tokens from pretrained embeddings to vocab.")
            assert pretrained_embeddings
            assert feat.min_freq == 1
            pretrained_vocab = {
                token
                for token, i in pretrained_embeddings.stoi.items()
                if feat.vocab_size <= 0 or i < feat.vocab_size
            }
            data.append([pretrained_vocab])

        return data

    def _gen_extra_metadata(self) -> None:
        """Subclass can overwrite to add more necessary metadata."""
        pass

    def get_train_iter_from_path(
        self, train_path: str, batch_size: int, rank: int = 0, world_size: int = 1
    ) -> BatchIterator:
        """
        Generate data batch iterator for training data. See `_get_train_iter()` for
        details

        Args:
            train_path (str): file path of training data
            batch_size (int): batch size
            rank (int): used for distributed training, the rank of current Gpu,
                don't set it to anything but 0 for non-distributed training
            world_size (int): used for distributed training, total number of Gpu
        """
        return self._get_train_iter(
            self.gen_dataset_from_path(train_path, rank=rank, world_size=world_size),
            batch_size,
            world_size,
        )

    def get_test_iter_from_path(self, test_path: str, batch_size: int) -> BatchIterator:
        return self._get_test_iter(self.gen_dataset_from_path(test_path), batch_size)

    def get_train_iter(self, rank: int = 0, world_size: int = 1):
        return self.get_train_iter_from_path(
            self.train_path, self.train_batch_size, rank, world_size
        )

    def get_eval_iter(self):
        return self.get_train_iter_from_path(self.eval_path, self.eval_batch_size)

    def get_test_iter(self):
        return self.get_test_iter_from_path(self.test_path, self.test_batch_size)

    def get_train_iter_from_raw_data(
        self,
        train_data: List[Dict[str, Any]],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> BatchIterator:
        shard_range = distributed.get_shard_range(len(train_data), rank, world_size)
        return self._get_train_iter(
            self.gen_dataset(train_data, shard_range=shard_range),
            batch_size,
            world_size,
        )

    def get_test_iter_from_raw_data(
        self, test_data: List[Dict[str, Any]], batch_size: int
    ) -> BatchIterator:
        return self._get_test_iter(self.gen_dataset(test_data), batch_size)

    def _get_train_iter(
        self, shard_dataset: textdata.Dataset, batch_size: int, world_size: int = 1
    ) -> BatchIterator:
        """
        Generate data batch iterator for training data. If distributed training
        is enabled, the dataset will be partitioned first. We use BucketIterator
        here to pool together examples with a similar size length to reduce the
        padding required for each batch.

        Args:
            shard_dataset (str): sharded training or evaluation dataset
            batch_size (int): batch size
            rank (int): used for distributed training, the rank of current Gpu,
                don't set it to anything but 0 for non-distributed training
            world_size (int): used for distributed training, total number of Gpu
        """
        # Compute the per-worker batch size
        batch_size = batch_size // world_size or batch_size

        return BatchIterator(
            textdata.BucketIterator(
                shard_dataset,
                batch_size=batch_size,
                device="cuda:{}".format(torch.cuda.current_device())
                if cuda.CUDA_ENABLED
                else "cpu",
                sort_within_batch=self.sort_within_batch,
                repeat=False,
                sort_key=self.sort_key,
                shuffle=self.shuffle,
            ),
            self._postprocess_batch,
            num_batches=math.ceil(len(shard_dataset) / float(batch_size)),
        )

    def _get_test_iter(
        self, test_dataset: textdata.Dataset, batch_size: int
    ) -> BatchIterator:
        return BatchIterator(
            textdata.Iterator(
                test_dataset,
                batch_size=batch_size,
                device="cuda:{}".format(torch.cuda.current_device())
                if cuda.CUDA_ENABLED
                else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
            is_train=False,
            num_batches=math.ceil(len(test_dataset) / float(batch_size)),
        )

    def get_predict_iter(
        self, data: Iterable[Dict[str, Any]], batch_size: Optional[int] = None
    ):
        ds = self.gen_dataset(data, include_label_fields=False)
        num_batches = (
            1 if batch_size is None else math.ceil(len(ds) / float(batch_size))
        )
        it = BatchIterator(
            textdata.Iterator(
                ds,
                batch_size=len(ds) if batch_size is None else batch_size,
                device="cuda:{}".format(torch.cuda.current_device())
                if cuda.CUDA_ENABLED
                else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
                sort_within_batch=self.sort_within_batch,
                shuffle=self.shuffle,
            ),
            self._postprocess_batch,
            include_target=False,
            is_train=False,
            num_batches=num_batches,
        )
        if batch_size is not None:
            return it
        else:
            for input, _, context in it:
                # only return the first batch since there is only one
                return input, context

    def read_from_file(
        self, file_name: str, columns_to_use: Union[Dict[str, int], List[str]]
    ) -> Generator[Dict, None, None]:
        """
        Read data from csv file. Input file format is required to be
        tab-separated columns

        Args:
            file_name (str): csv file name
            columns_to_use (Union[Dict[str, int], List[str]]): either a list of
                column names or a dict of column name -> column index in the file
        """
        print("reading data from {}".format(file_name))
        if isinstance(columns_to_use, list):
            columns_to_use = {
                name: idx
                for name, idx in zip(columns_to_use, range(len(columns_to_use)))
            }

        with open(file_name, "r", encoding="utf-8", errors="replace") as f_handle:
            csv_reader = csv.reader(f_handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            i = 0
            while True:
                i += 1
                try:
                    row = next(csv_reader)
                except csv.Error:
                    print("ignoring line {}".format(i))
                    continue
                except StopIteration:
                    break

                yield {
                    name: row[index] if index < len(row) else ""
                    for name, index in columns_to_use.items()
                }

    def _postprocess_batch(
        self,
        batch,
        include_input=True,
        include_target=True,
        include_context=True,
        is_train=True,
    ) -> Tuple:
        return (
            self._input_from_batch(batch, is_train) if include_input else None,
            self._target_from_batch(batch) if include_target else None,
            self._context_from_batch(batch) if include_context else None,
        )

    def _add_target_prob_to_res(self, res, row_data):
        if DFColumn.TARGET_PROBS in row_data:
            res[Target.TARGET_PROB_FIELD] = parse_json_array(
                row_data[DFColumn.TARGET_PROBS]
            )
        if DFColumn.TARGET_LABELS in row_data:
            res[Target.TARGET_LABEL_FIELD] = parse_json_array(
                row_data[DFColumn.TARGET_LABELS]
            )
        if DFColumn.TARGET_LOGITS in row_data:
            res[Target.TARGET_LOGITS_FIELD] = parse_json_array(
                row_data[DFColumn.TARGET_LOGITS]
            )

    def _target_from_batch(self, batch):
        targets = []
        for name in self.labels:
            target = getattr(batch, name)
            if name in [Target.TARGET_PROB_FIELD, Target.TARGET_LOGITS_FIELD]:
                label_vocab = self.metadata.target.vocab.stoi
                batch_label_list = getattr(batch, Target.TARGET_LABEL_FIELD)
                target = align_target_labels(target, batch_label_list, label_vocab)
            targets.append(target)
        if len(targets) == 1:
            return targets[0]
        return tuple(targets)

    def _input_from_batch(self, batch, is_train=True):
        return (
            self._train_input_from_batch(batch)
            if is_train
            else self._test_input_from_batch(batch)
        )

    def _train_input_from_batch(self, batch):
        return tuple(getattr(batch, name) for name in self.features)

    def _test_input_from_batch(self, batch):
        return self._train_input_from_batch(batch)

    def _context_from_batch(self, batch):
        return {
            name: getattr(batch, name)
            for name in self.extra_fields
            if hasattr(batch, name)
        }
