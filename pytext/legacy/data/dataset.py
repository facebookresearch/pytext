#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import gzip
import io
import os
import shutil
import tarfile
import zipfile
from functools import partial

import torch.utils.data
from torchtext.data.utils import RandomShuffler
from torchtext.utils import download_from_url

from .example import Example


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields.

    Attributes:
        sort_key (callable): A key to use for sorting dataset examples for batching
            together examples with similar lengths to minimize padding.
        examples (list(Example)): The examples in this dataset.
        fields (dict[str, Field]): Contains the name of each column or field, together
            with the corresponding Field object. Two fields with the same Field object
            will have a shared vocabulary.
    """

    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        self.examples = examples
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    @classmethod
    def splits(
        cls, path=None, root=".data", train=None, validation=None, test=None, **kwargs
    ):
        """Create Dataset objects for multiple splits of a dataset.

        Arguments:
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root (str): Root dataset storage directory. Default is '.data'.
            train (str): Suffix to add to path for the train set, or None for no
                train set. Default is None.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            Remaining keyword arguments: Passed to the constructor of the
                Dataset (sub)class being used.

        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if provided.
        """
        if path is None:
            path = cls.download(root)
        train_data = None if train is None else cls(os.path.join(path, train), **kwargs)
        val_data = (
            None
            if validation is None
            else cls(os.path.join(path, validation), **kwargs)
        )
        test_data = None if test is None else cls(os.path.join(path, test), **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

    def split(
        self, split_ratio=0.7, stratified=False, strata_field="label", random_state=None
    ):
        """Create train-test(-valid?) splits from the instance's examples.

        Arguments:
            split_ratio (float or List of floats): a number [0, 1] denoting the amount
                of data to be used for the training split (rest is used for test),
                or a list of numbers denoting the relative sizes of train, test and valid
                splits respectively. If the relative size for valid is missing, only the
                train-test split is returned. Default is 0.7 (for the train set).
            stratified (bool): whether the sampling should be stratified.
                Default is False.
            strata_field (str): name of the examples Field stratified over.
                Default is 'label' for the conventional label field.
            random_state (tuple): the random seed used for shuffling.
                A return value of `random.getstate()`.

        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if the splits are provided.
        """
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)

        # For the permutations
        rnd = RandomShuffler(random_state)
        if not stratified:
            train_data, test_data, val_data = rationed_split(
                self.examples, train_ratio, test_ratio, val_ratio, rnd
            )
        else:
            if strata_field not in self.fields:
                raise ValueError(
                    "Invalid field name for strata_field {}".format(strata_field)
                )
            strata = stratify(self.examples, strata_field)
            train_data, test_data, val_data = [], [], []
            for group in strata:
                # Stratify each group and add together the indices.
                group_train, group_test, group_val = rationed_split(
                    group, train_ratio, test_ratio, val_ratio, rnd
                )
                train_data += group_train
                test_data += group_test
                val_data += group_val

        splits = tuple(
            Dataset(d, self.fields) for d in (train_data, val_data, test_data) if d
        )

        # In case the parent sort key isn't none
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key
        return splits

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    @classmethod
    def download(cls, root, check=None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).

        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.

        Returns:
            str: Path to extracted dataset.
        """
        path = os.path.join(root, cls.name)
        check = path if check is None else check
        if not os.path.isdir(check):
            for url in cls.urls:
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print("downloading {}".format(filename))
                    download_from_url(url, zpath)
                zroot, ext = os.path.splitext(zpath)
                _, ext_inner = os.path.splitext(zroot)
                if ext == ".zip":
                    with zipfile.ZipFile(zpath, "r") as zfile:
                        print("extracting")
                        zfile.extractall(path)
                # tarfile cannot handle bare .gz files
                elif ext == ".tgz" or ext == ".gz" and ext_inner == ".tar":
                    with tarfile.open(zpath, "r:gz") as tar:
                        dirs = list(tar.getmembers())
                        tar.extractall(path=path, members=dirs)
                elif ext == ".gz":
                    with gzip.open(zpath, "rb") as gz:
                        with open(zroot, "wb") as uncompressed:
                            shutil.copyfileobj(gz, uncompressed)

        return os.path.join(path, cls.dirname)

    def filter_examples(self, field_names):
        """Remove unknown words from dataset examples with respect to given field.

        Arguments:
            field_names (list(str)): Within example only the parts with field names in
                field_names will have their unknown words deleted.
        """
        for i, example in enumerate(self.examples):
            for field_name in field_names:
                vocab = set(self.fields[field_name].vocab.stoi)
                text = getattr(example, field_name)
                example_part = [word for word in text if word in vocab]
                setattr(example, field_name, example_part)
            self.examples[i] = example


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(
        self, path, format, fields, skip_header=False, csv_reader_params=None, **kwargs
    ):
        """Create a TabularDataset given a path, file format, and field list.

        Args:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields ((list(tuple(str, Field)) or dict[str: tuple(str, Field)): If using a list,
                the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
            csv_reader_params(dict): Parameters to pass to the csv reader.
                Only relevant when format is csv or tsv.
                See
                https://docs.python.org/3/library/csv.html#csv.reader
                for more details.
            kwargs (dict): passed to the Dataset parent class.
        """
        if csv_reader_params is None:
            csv_reader_params = {}
        format = format.lower()
        make_example = {
            "json": Example.fromJSON,
            "dict": Example.fromdict,
            "tsv": Example.fromCSV,
            "csv": Example.fromCSV,
        }[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == "csv":
                reader = csv.reader(f, **csv_reader_params)
            elif format == "tsv":
                reader = csv.reader(f, delimiter="\t", **csv_reader_params)
            else:
                reader = f

            if format in ["csv", "tsv"] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError(
                        "When using a dict to specify fields with a {} file,"
                        "skip_header must be False and"
                        "the file must have a header.".format(format)
                    )
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)


def check_split_ratio(split_ratio):
    """Check that the split ratio argument is not malformed"""
    valid_ratio = 0.0
    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        # Assert in bounds, validation size is zero
        assert 0.0 < split_ratio < 1.0, "Split ratio {} not between 0 and 1".format(
            split_ratio
        )

        test_ratio = 1.0 - split_ratio
        return (split_ratio, test_ratio, valid_ratio)
    elif isinstance(split_ratio, list):
        # A list of relative ratios is provided
        length = len(split_ratio)
        assert (
            length == 2 or length == 3
        ), "Length of split ratio list should be 2 or 3, got {}".format(split_ratio)

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.0:
            split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError(
            "Split ratio must be float or a list, got {}".format(type(split_ratio))
        )


def stratify(examples, strata_field):
    # The field has to be hashable otherwise this doesn't work
    # There's two iterations over the whole dataset here, which can be
    # reduced to just one if a dedicated method for stratified splitting is used
    unique_strata = {getattr(example, strata_field) for example in examples}
    strata_maps = {s: [] for s in unique_strata}
    for example in examples:
        strata_maps[getattr(example, strata_field)].append(example)
    return list(strata_maps.values())


def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
    """Create a random permutation of examples, then split them by ratios

    Arguments:
        examples: a list of data
        train_ratio, test_ratio, val_ratio: split fractions.
        rnd: a random shuffler

    Examples:
        >>> examples = []
        >>> train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
        >>> rnd = torchtext.data.dataset.RandomShuffler(None)
        >>> train_examples, test_examples, valid_examples = \
                torchtext.data.dataset.rationed_split(examples, train_ratio,
                                                      test_ratio, val_ratio,
                                                      rnd)
    """
    N = len(examples)
    randperm = rnd(range(N))
    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if not val_ratio:
        test_len = N - train_len
    else:
        test_len = int(round(test_ratio * N))

    indices = (
        randperm[:train_len],  # Train
        randperm[train_len : train_len + test_len],  # Test
        randperm[train_len + test_len :],
    )  # Validation

    # There's a possibly empty list for the validation set
    data = tuple([examples[i] for i in index] for index in indices)

    return data
