#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from enum import Enum
from typing import List, NamedTuple, Optional

from pytext.utils.path import PYTEXT_HOME


TEST_DATA_DIR = os.environ.get(
    "PYTEXT_TEST_DATA", os.path.join(PYTEXT_HOME, "tests/data")
)


TEST_CONFIG_DIR = os.environ.get(
    "PYTEXT_TEST_CONFIG", os.path.join(PYTEXT_HOME, "demo/configs")
)


def test_file(filename):
    return os.path.join(TEST_DATA_DIR, filename)


class TestFileName(Enum):
    def __str__(self):
        return str(self.value)

    TRAIN_DENSE_FEATURES_TINY_TSV = "train_dense_features_tiny.tsv"
    TEST_PERSONALIZATION_OPPOSITE_INPUTS_TSV = (
        "test_personalization_opposite_inputs.tsv"
    )
    TEST_PERSONALIZATION_SAME_INPUTS_TSV = "test_personalization_same_inputs.tsv"
    TEST_PERSONALIZATION_SINGLE_USER_TSV = "test_personalization_single_user.tsv"


class TestFileMetadata(NamedTuple):
    filename: str
    field_names: Optional[List[str]] = None
    dense_col_name: Optional[str] = None
    dense_feat_dim: Optional[int] = None
    uid_col_name: Optional[str] = None


TEST_FILE_NAME_TO_METADATA = {
    TestFileName.TRAIN_DENSE_FEATURES_TINY_TSV: TestFileMetadata(
        filename=test_file(str(TestFileName.TRAIN_DENSE_FEATURES_TINY_TSV)),
        field_names=["label", "slots", "text", "dense_features"],
        dense_col_name="dense_features",
        dense_feat_dim=10,
    ),
    TestFileName.TEST_PERSONALIZATION_OPPOSITE_INPUTS_TSV: TestFileMetadata(
        filename=test_file(str(TestFileName.TEST_PERSONALIZATION_OPPOSITE_INPUTS_TSV)),
        field_names=["label", "text", "dense_features", "uid"],
        dense_col_name="dense_features",
        dense_feat_dim=10,
        uid_col_name="uid",
    ),
    TestFileName.TEST_PERSONALIZATION_SAME_INPUTS_TSV: TestFileMetadata(
        filename=test_file(str(TestFileName.TEST_PERSONALIZATION_SAME_INPUTS_TSV)),
        field_names=["label", "text", "dense_features", "uid"],
        dense_col_name="dense_features",
        dense_feat_dim=10,
        uid_col_name="uid",
    ),
    TestFileName.TEST_PERSONALIZATION_SINGLE_USER_TSV: TestFileMetadata(
        filename=test_file(str(TestFileName.TEST_PERSONALIZATION_SINGLE_USER_TSV)),
        field_names=["label", "text", "dense_features", "uid"],
        dense_col_name="dense_features",
        dense_feat_dim=10,
        uid_col_name="uid",
    ),
}


def get_test_file_metadata(test_file_id: TestFileName) -> TestFileMetadata:
    return TEST_FILE_NAME_TO_METADATA[test_file_id]
