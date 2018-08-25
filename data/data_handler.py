#!/usr/bin/env python3
import csv
from copy import copy
from typing import Any, Dict, Generator, List, Tuple, Type

import pandas as pd
import torch
from pytext.common.constants import VocabMeta
from pytext.config.component import Component, ComponentType
from pytext.config.field_config import EmbedInitStrategy
from pytext.fields import Field, FieldMeta
from pytext.utils import cuda_utils, embeddings_utils
from torchtext import data as textdata


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
        raw_columns: columns to read from file and put into pandas dataframe, the order
            should be the same as how data stored in tsv file
        labels:
        features:
        df_to_feat_func_map: a map defines how to convert a pandas dataframe column
            to raw feature to construct torchtext Example, each item can be:
            1 Nothing, then column name will be the same as feature name
            2 feat_name -> column_name
            3 feat_name -> function, take one row as input and output the feature
    """

    __COMPONENT_TYPE__ = ComponentType.DATA_HANDLER

    # special df index field to record initial position of examples
    DF_INDEX = "__df_index"

    # TODO exposing sort_key leaks the implementation detail that torchtext Example
    # is used, need to think about it if we want the interface decoupled with torchtext
    def __init__(
        self,
        raw_columns: List[str],
        labels: List[Field],
        features: List[Field],
        extra_fields: List[Field] = None,
        pretrained_embeds_file: str = None,
        embed_dim: int = 0,
        embed_init_strategy: EmbedInitStrategy = EmbedInitStrategy.RANDOM,
    ) -> None:
        self.raw_columns: List[str] = raw_columns or []
        self.labels: List[Field] = labels or []
        self.features: List[Field] = features or []
        self.extra_fields: List[Field] = extra_fields or []
        # Presume the first feature should be the text feature
        self.text_field = self.features[0]
        self.pretrained_embeds_file = pretrained_embeds_file
        self.embed_dim = embed_dim
        self.embed_init_strategy = embed_init_strategy
        self.df_to_feat_func_map: Dict = {}
        self.metadata_cls: Type = CommonMetadata
        self.metadata: CommonMetadata = CommonMetadata()
        self._data_cache: Dict = {}

    def sort_key(self, ex: textdata.Example) -> Any:
        return len(getattr(ex, self.text_field.name))

    def metadata_to_save(self):
        # make a copy
        metadata = copy(self.metadata)
        # pretrained_embeds_weight takes a lot space and is not needed in inference time
        metadata.pretrained_embeds_weight = None
        return metadata

    def load_metadata(self, metadata: CommonMetadata):
        self.metadata = metadata
        for f in self.features:
            if f.use_vocab and f.name in metadata.features:
                f.vocab = metadata.features[f.name].vocab
        for f in self.labels:
            if f.use_vocab and f.name in metadata.labels:
                f.vocab = metadata.labels[f.name].vocab

    def gen_dataset_from_file(
        self, file_name: str, include_label_fields: bool = True, use_cache: bool = True
    ) -> textdata.Dataset:
        if use_cache and file_name in self._data_cache:
            return self._data_cache[file_name]
        res = self.gen_dataset(
            self.read_from_file(file_name, self.raw_columns), include_label_fields
        )
        self._data_cache[file_name] = res
        return res

    def gen_dataset(
        self, df: pd.DataFrame, include_label_fields: bool = True
    ) -> textdata.Dataset:
        """ generate torchtext Dataset from dataframe.
        """
        # preprocess df
        df = self._preprocess_df(df)

        to_process = self.features + self.extra_fields
        if include_label_fields:
            to_process = self.labels + to_process
        # define torch text fields
        fields = [(feat.name, feat) for feat in to_process]
        # generate example from dataframe
        all_examples = [
            textdata.Example.fromlist(row, fields)
            for row in self._df_to_field(df, to_process)
        ]
        res = textdata.Dataset(all_examples, fields)
        # extra context
        for k, v in self._gen_dataset_context(df).items():
            setattr(res, k, v)
        return res

    def _gen_dataset_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def init_metadata_from_file(self, train_file_path, eval_file_path, test_file_path):
        # get data sets
        self._init_metadata(
            *[
                self.gen_dataset_from_file(file_path)
                for file_path in [train_file_path, eval_file_path, test_file_path]
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
        for label in self.labels:
            # TODO shall we only use train and eval for non-bio labels?
            # Need test data to make sure we cover all of the labels in it
            # It is particularly important when BIO is enabled as a B-[Label] can
            # appear in train and eval but test can have B-[Label] and I-[Label]
            if label.use_vocab:
                print("building vocab for label {}".format(label.name))
                label.build_vocab(train_data, eval_data, test_data)

        # build vocabs for features
        for feat in self.features:
            if feat.use_vocab:
                print("building vocab for feature {}".format(feat.name))
                feat.build_vocab(train_data)

        # field metadata
        self.metadata.features = {f.name: f.get_meta() for f in self.features}
        self.metadata.labels = {f.name: f.get_meta() for f in self.labels}

        # pretrained embedding weight
        if self.pretrained_embeds_file:
            weight = embeddings_utils.init_pretrained_embeddings(
                self.text_field.vocab.stoi,
                self.pretrained_embeds_file,
                self.embed_dim,
                VocabMeta.UNK_TOKEN,
                init_strategy=self.embed_init_strategy,
            )
            self.metadata.pretrained_embeds_weight = weight

        self._gen_extra_metadata()

    def _gen_extra_metadata(self) -> None:
        """Subclass can overwrite to add more necessary metadata
        """
        pass

    def get_train_batch_from_file(
        self, file_names: Tuple[str, ...], batch_size: Tuple[int, ...]
    ) -> Tuple[BatchIterator, ...]:
        return self._get_train_batch(
            tuple(self.gen_dataset_from_file(f) for f in file_names), batch_size
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
            )
        )

    def get_test_batch(self, file_path: str, batch_size: int) -> BatchIterator:
        test_data = self.gen_dataset_from_file(file_path)
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
        """ Read data from file and generate a dataframe to host intermediate dataself.
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

    def _postprocess_batch(
        self, batch, include_input=True, include_target=True, include_context=True
    ) -> Tuple:
        return (
            self._input_from_batch(batch) if include_input else None,
            self._target_from_batch(batch) if include_target else None,
            self._context_from_batch(batch) if include_context else None,
        )

    def _target_from_batch(self, batch):
        return tuple(getattr(batch, label.name) for label in self.labels)

    def _input_from_batch(self, batch):
        return tuple(getattr(batch, feat.name) for feat in self.features)

    def _context_from_batch(self, batch):
        return {f.name: getattr(batch, f.name) for f in self.extra_fields}

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            preprocess the dataframe read from file, generate some intermediate data
            do nothing by default
        """
        pass

    def _df_to_field(
        self, df, to_process: List[Field]
    ) -> Generator[List[Any], None, None]:
        for idx, row in df.iterrows():
            yield [self._apply_map_func(field, row, idx) for field in to_process]

    def _apply_map_func(self, field, row, idx):
        name = field.name
        row_to_field = self.df_to_feat_func_map.get(name)
        # try to get the feature with same name in df if no mapping method is defined
        if row_to_field is None:
            return row.get(name, None)
        # using label
        if type(row_to_field) is str:
            if row_to_field == self.DF_INDEX:
                return idx
            return row.get(row_to_field, None)
        # map function
        else:
            return row_to_field(row, field)
