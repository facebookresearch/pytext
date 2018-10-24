#!/usr/bin/env python3
import csv
import math
import multiprocessing
import os
from copy import copy, deepcopy
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple, Type, Union

import torch
from pytext.common.constants import BatchContext, DatasetFieldName, VocabMeta
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.data.featurizer import Featurizer
from pytext.fields import Field, FieldMeta, VocabUsingField
from pytext.utils import cuda_utils, embeddings_utils
from torchtext import data as textdata


class CommonMetadata:
    features: Dict[str, FieldMeta]
    labels: Dict[str, FieldMeta]
    pretrained_embeds_weight: Optional[torch.Tensor] = None
    seq_pretrained_embeds_weight: Optional[torch.Tensor] = None


class BatchIterator:
    """
    Each iteration will return a tuple of (input, target, context)
    input can be feeded directly into model forward function
    target is the numaricalized label(s)
    context is any extra info to be used in downstream steps
    """

    def __init__(
        self,
        batches,
        processor,
        include_input=True,
        include_target=True,
        include_context=True,
        is_train=True,
        num_batches=-1,
    ):
        self.processor = processor
        self.batches = batches
        self.include_input = include_input
        self.include_target = include_target
        self.include_context = include_context
        self.is_train = is_train
        self.total_num_batches = num_batches

    def __iter__(self):
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
            # Due to a limitation in PyTroch's distributed training backend that
            # enforces that all the parallel workers to have the same number of
            # batches we keep yielding the last batch until the requested total
            # number of batches is fullfilled
            if i == num_batches - 1:
                context = deepcopy(context)
                context.update({BatchContext.IGNORE_LOSS: True})
                for _j in range(num_batches, self.total_num_batches):
                    yield (input, target, context)


class DataHandler(Component):
    """ Defines a pipeline to process data and generate tensors to be consumed by model,
        1 construct the DataHandler object
        2 call init_vocab to build necessary vocabs
        3 call get_meta to get meta data of all fields
        4 call batch to get batch iterator (check gen_dataset
          function to understand the details of the process pipeline)
        5 each batch is a (input, target, context) tuple, in which input can be feed
          directly into model.

    Attributes:
        raw_columns: columns to read from data source. In case of files, the order
            should match the data stored in that file
        labels:
        features:
        extra_fields
    """

    class Config(ConfigBase):
        columns_to_read: List[str] = []
        shuffle: bool = True
        train_path: str = "train.tsv"
        eval_path: str = "eval.tsv"
        test_path: str = "test.tsv"
        train_batch_size: int = 128
        eval_batch_size: int = 128
        test_batch_size: int = 128

    __COMPONENT_TYPE__ = ComponentType.DATA_HANDLER

    # special df index field to record initial position of examples
    DF_INDEX = "__df_index"

    def __init__(
        self,
        raw_columns: List[str],
        labels: Dict[str, Field],
        features: Dict[str, Field],
        featurizer: Featurizer,
        extra_fields: Dict[str, Field] = None,
        text_feature_name: str = DatasetFieldName.TEXT_FIELD,
        shuffle: bool = True,
        train_path: str = "train.tsv",
        eval_path: str = "eval.tsv",
        test_path: str = "test.tsv",
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        test_batch_size: int = 128,
        max_seq_len: int = -1,
    ) -> None:
        self.raw_columns: List[str] = raw_columns or []
        self.labels: Dict[str, Field] = labels or {}
        self.features: Dict[str, Field] = features or {}
        self.featurizer = featurizer
        self.extra_fields: Dict[str, Field] = extra_fields or {}
        self.text_feature_name: str = text_feature_name

        self.df_to_example_func_map: Dict = {}
        self.metadata_cls: Type = CommonMetadata
        self.metadata: CommonMetadata = CommonMetadata()
        self._data_cache: MutableMapping[str, Any] = {}
        self.shuffle = shuffle
        self.num_workers = multiprocessing.cpu_count()
        self.max_seq_len = max_seq_len

        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size

    def load_vocab(self, vocab_file, vocab_size, lowercase_tokens: bool = False):
        """
        Loads items into a set from a file containing one item per line.
        Items are added to the set from top of the file to bottom.
        So, the items in the file should be ordered by a preference (if any), e.g.,
        it makes sense to order tokens in descending order of frequency in corpus.
        """
        vocab: Set[str] = set()
        if os.path.isfile(vocab_file):
            with open(vocab_file, "r") as f:
                for i, line in enumerate(f):
                    if len(vocab) == vocab_size:
                        print(
                            f"Read {i+1} items from {vocab_file}"
                            f"to load vocab of size {vocab_size}."
                            f"Skipping rest of the file"
                        )
                        break
                    line = line.strip()
                    vocab.add(line.lower() if lowercase_tokens else line)
        else:
            print(f"{vocab_file} doesn't exist. Cannot load vocabulary from it")
        return vocab

    def sort_key(self, ex: textdata.Example) -> Any:
        return len(getattr(ex, self.text_feature_name))

    def metadata_to_save(self):
        # make a copy
        metadata = copy(self.metadata)
        # pretrained_embeds_weight takes a lot space and is not needed in inference time
        metadata.pretrained_embeds_weight = None
        return metadata

    def load_metadata(self, metadata: CommonMetadata):
        self.metadata = metadata
        for name, field in self.features.items():
            if field.use_vocab and name in metadata.features:
                field.load_meta(metadata.features[name])
        for name, field in self.labels.items():
            if field.use_vocab and name in metadata.labels:
                field.load_meta(metadata.labels[name])

    def gen_dataset_from_path(
        self, path: str, include_label_fields: bool = True, use_cache: bool = True
    ) -> textdata.Dataset:
        if use_cache and path in self._data_cache:
            return self._data_cache[path]
        res = self.gen_dataset(
            self.read_from_file(path, self.raw_columns), include_label_fields
        )
        self._data_cache[path] = res
        return res

    def gen_dataset(
        self, data: List[Dict[str, Any]], include_label_fields: bool = True
    ) -> textdata.Dataset:
        """ generate torchtext Dataset from dataframe.
        """
        to_process = {}
        to_process.update(self.features)
        to_process.update(self.extra_fields)
        if include_label_fields:
            to_process.update(self.labels)
        fields = {name: (name, field) for name, field in to_process.items()}
        # generate example from dataframe
        examples = [
            textdata.Example.fromdict(row, fields) for row in self.preprocess(data)
        ]
        return textdata.Dataset(examples, to_process)

    def preprocess(self, data: List[Dict[str, Any]]):
        for idx, row in enumerate(data):
            yield self.preprocess_row(row, idx)

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """
        preprocess steps for a single input row
        """
        return row_data

    def init_metadata_from_path(self, train_path, eval_path, test_path):
        # get data sets
        self._init_metadata(
            *[
                self.gen_dataset_from_path(path)
                for path in [train_path, eval_path, test_path]
            ]
        )

    def init_metadata(self):
        self.init_metadata_from_path(self.train_path, self.eval_path, self.test_path)

    def init_metadata_from_raw_data(self, *data):
        self._init_metadata(*[self.gen_dataset(d) for d in data])

    def _init_metadata(
        self,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ):
        # build vocabs for label fields
        for name, label in self.labels.items():
            # Need test data to make sure we cover all of the labels in it
            # It is particularly important when BIO is enabled as a B-[Label] can
            # appear in train and eval but test can have B-[Label] and I-[Label]

            # if vocab is already built, skip
            if label.use_vocab and not getattr(label, "vocab", None):
                print("building vocab for label {}".format(name))
                label.build_vocab(train_data, eval_data, test_data)
                print(
                    "{} field's vocabulary size is {}".format(
                        name, len(label.vocab.itos)
                    )
                )

        # build vocabs for features
        for name, feat in self.features.items():
            if feat.use_vocab:
                print("building vocab for feature {}".format(name))
                feat.build_vocab(
                    *self._get_data_to_build_vocab(
                        feat, train_data, eval_data, test_data
                    )
                )
                print("{} field's vocabulary size is {}".format(name, len(feat.vocab)))

                # Initialize pretrained embedding weights.
                if (
                    hasattr(feat, "pretrained_embeddings_path")
                    and feat.pretrained_embeddings_path
                ):
                    weights = embeddings_utils.init_pretrained_embeddings(
                        feat.vocab.stoi,
                        feat.pretrained_embeddings_path,
                        feat.embed_dim,
                        VocabMeta.UNK_TOKEN,
                        feat.embedding_init_strategy,
                        feat.lower,
                    )
                    if name == DatasetFieldName.TEXT_FIELD:
                        self.metadata.pretrained_embeds_weight = weights
                    if name == DatasetFieldName.SEQ_FIELD:
                        self.metadata.seq_pretrained_embeds_weight = weights

        # field metadata
        self.metadata.features = {
            name: field.get_meta() for name, field in self.features.items()
        }
        self.metadata.labels = {
            name: field.get_meta() for name, field in self.labels.items()
        }

        self._gen_extra_metadata()

    def _get_data_to_build_vocab(
        self,
        feat: Field,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ) -> List[Any]:
        """
        This method prepares the list of data sources that
        Field.build_vocab() accepts to build vocab from.

        If vocab building from training data is configured then that Dataset
        object is appended to build vocabulary from.

        If a vocab file is provided an additional data source built from the file
        is appended to the list which, is a list of items read from the vocab file.
        """
        data = []
        if isinstance(feat, VocabUsingField):
            if feat.vocab_from_train_data:
                data.append(train_data)
            elif feat.vocab_from_all_data:
                data.extend([train_data, eval_data, test_data])
        if hasattr(feat, "vocab_file") and feat.vocab_file:
            lowercase_tokens = feat.lower if hasattr(feat, "lower") else False
            data.append(
                [self.load_vocab(feat.vocab_file, feat.vocab_size, lowercase_tokens)]
            )
        return data

    def _gen_extra_metadata(self) -> None:
        """Subclass can overwrite to add more necessary metadata."""
        pass

    def get_train_iter_from_path(
        self, train_path: str, batch_size: int, rank: int = 0, world_size: int = 1
    ) -> BatchIterator:
        return self._get_train_iter(
            self.gen_dataset_from_path(train_path), batch_size, rank, world_size
        )

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
        return self._get_train_iter(
            self.gen_dataset(train_data), batch_size, rank, world_size
        )

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
            textdata.BucketIterator(
                dataset_shard,
                batch_size=batch_size,
                device="cuda:{}".format(torch.cuda.current_device())
                if cuda_utils.CUDA_ENABLED
                else "cpu",
                sort_within_batch=True,
                repeat=False,
                sort_key=self.sort_key,
                shuffle=self.shuffle,
            ),
            self._postprocess_batch,
            num_batches=math.ceil(num_all_batches / float(world_size)),
        )

    def get_test_iter_from_path(self, path: str, batch_size: int) -> BatchIterator:
        test_data = self.gen_dataset_from_path(path)
        return BatchIterator(
            textdata.Iterator(
                test_data,
                batch_size=batch_size,
                device="cuda:{}".format(torch.cuda.current_device())
                if cuda_utils.CUDA_ENABLED
                else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
            is_train=False,
        )

    def get_predict_iter(self, data: List[Dict[str, Any]]):
        ds = self.gen_dataset(data, include_label_fields=False)
        it = BatchIterator(
            textdata.Iterator(
                ds,
                batch_size=len(ds),
                device="cuda:{}".format(torch.cuda.current_device())
                if cuda_utils.CUDA_ENABLED
                else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
            include_target=False,
            is_train=False,
        )
        for input, _, context in it:
            # only return the first batch since there is only one
            return input, context

    @staticmethod
    def _get_dataset_shard(
        dataset: textdata.Dataset, rank: int, world_size: int
    ) -> textdata.Dataset:
        assert rank > -1 and world_size > 0
        shard_len = len(dataset) // world_size
        shard_offset = rank * shard_len
        shard_end = len(dataset) if rank == world_size - 1 else shard_offset + shard_len
        return textdata.Dataset(
            dataset.examples[shard_offset:shard_end], dataset.fields
        )

    @staticmethod
    def read_from_file(
        file_name: str, columns_to_use: Union[Dict[str, int], List[str]]
    ) -> List[Dict[str, Any]]:
        """ Read data from csv file. Input file format is required to be
        tab-separated columns
        """
        print("reading data from {}".format(file_name))
        if isinstance(columns_to_use, list):
            columns_to_use = {
                name: idx
                for name, idx in zip(columns_to_use, range(len(columns_to_use)))
            }
        with open(file_name, "r", encoding="utf-8", errors="replace") as f_handle:
            csv_reader = csv.reader(f_handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            data = []
            for row in csv_reader:
                row_len = len(row)
                row_data = {}
                for name, idx in columns_to_use.items():
                    value = row[idx] if idx < row_len else ""
                    row_data[name] = value
                data.append(row_data)
            return data

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

    def _target_from_batch(self, batch):
        targets = tuple(getattr(batch, name) for name in self.labels)
        if len(targets) == 1:
            return targets[0]
        return targets

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
        return {name: getattr(batch, name) for name in self.extra_fields}
