#!/usr/bin/env python3
import csv
import multiprocessing
import os
from copy import copy
from typing import Any, Dict, Generator, List, Set, Tuple, Type

import pandas as pd
import torch
import torch.hiveio as hiveio
from pytext.common.constants import DatasetFieldName, VocabMeta
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.fields import Field, FieldMeta, VocabUsingField
from pytext.utils import cuda_utils, embeddings_utils
from torchtext import data as textdata


# Special prefix to distnguish the hive data source
HIVE_PREFIX = "hive://"


class CommonMetadata:
    features: Dict[str, FieldMeta]
    labels: Dict[str, FieldMeta]
    pretrained_embeds_weight: torch.Tensor = None


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
    ):
        self.processor = processor
        self.batches = batches
        self.include_input = include_input
        self.include_target = include_target
        self.include_context = include_context

    def __iter__(self):
        for batch in self.batches:
            yield self.processor(
                batch, self.include_input, self.include_target, self.include_context
            )


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
        raw_columns: columns to read from data source and put into pandas dataframe,
            in case of files; the order should match the data stored in that file
        labels:
        features:
        df_to_example_func_map: a map defines how to convert a pandas dataframe
        column to a column in torchtext Example, each item can be:
            1 Nothing, will pick the column in df with the field name
            2 field_name -> column_name, pick the column in df with the column name
            3 field_name -> function, take one row as input and output the processed
              data
    """

    class Config(ConfigBase):
        columns_to_read: List[str] = []
        shuffle: bool = True
        pretrained_embeddings_path = ""

    __COMPONENT_TYPE__ = ComponentType.DATA_HANDLER

    # special df index field to record initial position of examples
    DF_INDEX = "__df_index"

    def __init__(
        self,
        raw_columns: List[str],
        labels: Dict[str, Field],
        features: Dict[str, Field],
        extra_fields: Dict[str, Field] = None,
        text_feature_name: str = DatasetFieldName.TEXT_FIELD,
        shuffle: bool = True,
    ) -> None:
        self.raw_columns: List[str] = raw_columns or []
        self.labels: Dict[str, Field] = labels or {}
        self.features: Dict[str, Field] = features or {}
        self.extra_fields: Dict[str, Field] = extra_fields or {}
        self.text_feature_name: str = text_feature_name

        self.df_to_example_func_map: Dict = {}
        self.metadata_cls: Type = CommonMetadata
        self.metadata: CommonMetadata = CommonMetadata()
        self._data_cache: Dict = {}
        self.shuffle = (shuffle,)
        self.num_workers = multiprocessing.cpu_count()

    def load_vocab(self, vocab_file, vocab_size):
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
                    vocab.add(line.strip())
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
                field.vocab = metadata.features[name].vocab
        for name, field in self.labels.items():
            if field.use_vocab and name in metadata.labels:
                field.vocab = metadata.labels[name].vocab

    def gen_dataset_from_path(
        self, path: str, include_label_fields: bool = True, use_cache: bool = True
    ) -> textdata.Dataset:
        if use_cache and path in self._data_cache:
            return self._data_cache[path]
        res = self.gen_dataset(
            self.read_from_hive(path, self.raw_columns)
            if path.startswith(HIVE_PREFIX)
            else self.read_from_file(path, self.raw_columns),
            include_label_fields,
        )
        self._data_cache[path] = res
        return res

    def gen_dataset(
        self, df: pd.DataFrame, include_label_fields: bool = True
    ) -> textdata.Dataset:
        """ generate torchtext Dataset from dataframe.
        """
        # preprocess df
        df = self._preprocess_df(df)

        to_process = {}
        to_process.update(self.features)
        to_process.update(self.extra_fields)
        if include_label_fields:
            to_process.update(self.labels)
        # generate example from dataframe
        all_examples = [
            textdata.Example.fromlist(
                row, [(name, field) for name, field in to_process.items()]
            )
            for row in self._df_to_examples(df, to_process)
        ]
        res = textdata.Dataset(all_examples, to_process)
        # extra context
        for k, v in self._gen_dataset_context(df).items():
            setattr(res, k, v)
        return res

    def _gen_dataset_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def init_metadata_from_path(self, train_path, eval_path, test_path):
        # get data sets
        self._init_metadata(
            *[
                self.gen_dataset_from_path(path)
                for path in [train_path, eval_path, test_path]
            ]
        )

    def init_metadata_from_df(
        self, train_data: pd.DataFrame, eval_data: pd.DataFrame, test_data: pd.DataFrame
    ):
        self._init_metadata(
            *[self.gen_dataset(df) for df in [train_data, eval_data, test_data]]
        )

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
                feat.build_vocab(*self._get_data_to_build_vocab(feat, train_data))
                print(
                    "{} field's vocabulary size is {}".format(
                        name, len(feat.vocab.itos)
                    )
                )

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
                        embedding_init_strategy=feat.embedding_init_strategy,
                    )
                    self.metadata.pretrained_embeds_weight = weights

        # field metadata
        self.metadata.features = {
            name: field.get_meta() for name, field in self.features.items()
        }
        self.metadata.labels = {
            name: field.get_meta() for name, field in self.labels.items()
        }

        self._gen_extra_metadata()

    def _get_data_to_build_vocab(
        self, feat: Field, train_data: textdata.Dataset
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
        if isinstance(feat, VocabUsingField) and feat.vocab_from_train_data:
            data.append(train_data)
        if hasattr(feat, "vocab_file"):
            data.append([self.load_vocab(feat.vocab_file, feat.vocab_size)])
        return data

    def _gen_extra_metadata(self) -> None:
        """Subclass can overwrite to add more necessary metadata."""
        pass

    def get_train_batch_from_path(
        self, data_paths: Tuple[str, ...], batch_size: Tuple[int, ...]
    ) -> Tuple[BatchIterator, ...]:
        return self._get_train_batch(
            tuple(self.gen_dataset_from_path(p) for p in data_paths), batch_size
        )

    def get_train_batch_from_df(
        self, data_frames: Tuple[pd.DataFrame, ...], batch_size: Tuple[int, ...]
    ) -> Tuple[BatchIterator, ...]:
        return self._get_train_batch(
            tuple(self.gen_dataset(df) for df in data_frames), batch_size
        )

    def _get_train_batch(
        self, datasets: Tuple[textdata.Dataset, ...], batch_size: Tuple[int, ...]
    ) -> Tuple[BatchIterator, ...]:
        return tuple(
            BatchIterator(iter, self._postprocess_batch)
            for iter in textdata.BucketIterator.splits(
                datasets,
                batch_sizes=batch_size,
                device="cuda:0" if cuda_utils.CUDA_ENABLED else "cpu",
                sort_within_batch=True,
                repeat=False,
                sort_key=self.sort_key,
                shuffle=self.shuffle,
            )
        )

    def get_test_batch(self, path: str, batch_size: int) -> BatchIterator:
        test_data = self.gen_dataset_from_path(path)
        return BatchIterator(
            textdata.Iterator(
                test_data,
                batch_size=batch_size,
                device="cuda:0" if cuda_utils.CUDA_ENABLED else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
        )

    def get_predict_batch(self, df: pd.DataFrame):
        data = self.gen_dataset(df, include_label_fields=False)

        it = BatchIterator(
            textdata.Iterator(
                data,
                batch_size=len(data.examples),
                device="cuda:0" if cuda_utils.CUDA_ENABLED else "cpu",
                sort=True,
                repeat=False,
                train=False,
                sort_key=self.sort_key,
            ),
            self._postprocess_batch,
            include_target=False,
        )
        for input, _, context in it:
            # only return the first batch since there is only one
            return input, context

    @staticmethod
    def read_from_file(file_name: str, columns: List[str]) -> pd.DataFrame:
        """ Read data from file and generate a dataframe to host intermediate data.
            Input file format is required to be tab-separated columns
        """
        print("reading data from {}".format(file_name))
        # Replace characters with encoding errors
        # Doing replace instead of ignore to not cause alignment issues
        with open(file_name, "r", encoding="utf-8", errors="replace") as f_handle:
            return pd.read_csv(
                f_handle,
                header=None,
                encoding="utf-8",
                sep="\t",
                delim_whitespace=False,
                na_values="\\n",
                keep_default_na=False,
                dtype=str,
                quoting=csv.QUOTE_NONE,
                names=columns,
                index_col=False,
            )

    @staticmethod
    def read_from_hive(hive_path: str, columns: List[str]) -> pd.DataFrame:
        """ Read data from hive path in this format:
            hive://[namespace]/[table_name]/[partition_list]
            and generate a dataframe to host intermediate data.
        """
        print("reading data from hive path: {}".format(hive_path))
        assert hive_path.startswith(HIVE_PREFIX), "Invalid hive path: {}".format(
            hive_path
        )
        namespace, table, partitions = hive_path[len(HIVE_PREFIX) :].split("/", 3)
        partitions_list = partitions.split("/")
        col_list = hiveio.read(namespace, table, partitions_list, columns)
        return pd.DataFrame.from_dict(dict(zip(columns, col_list)))

    def _postprocess_batch(
        self, batch, include_input=True, include_target=True, include_context=True
    ) -> Tuple:
        return (
            self._input_from_batch(batch) if include_input else None,
            self._target_from_batch(batch) if include_target else None,
            self._context_from_batch(batch) if include_context else None,
        )

    def _target_from_batch(self, batch):
        return tuple(getattr(batch, name) for name in self.labels)

    def _input_from_batch(self, batch):
        return tuple(getattr(batch, name) for name in self.features)

    def _context_from_batch(self, batch):
        return {name: getattr(batch, name) for name in self.extra_fields}

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataframe read from path, generate some intermediate data.
        Do nothing by default.
        """
        pass

    def _df_to_examples(
        self, df, columns_to_process: Dict[str, Field]
    ) -> Generator[List[Any], None, None]:
        for idx, row in df.iterrows():
            yield [
                self._df_column_to_example_column(name, field, row, idx)
                for name, field in columns_to_process.items()
            ]

    def _df_column_to_example_column(self, field_name, field, row, idx):
        mapping_func = self.df_to_example_func_map.get(field_name)
        # try to get the feature with same name in df if no mapping method is defined
        if mapping_func is None:
            return row.get(field_name, None)
        # using a different column name
        if type(mapping_func) is str:
            column_name = mapping_func
            if column_name == self.DF_INDEX:
                return idx
            return row.get(column_name, None)
        # map function
        else:
            return mapping_func(row, field)
