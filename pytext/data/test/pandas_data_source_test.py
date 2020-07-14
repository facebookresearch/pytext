#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import pandas as pd
from pytext.config.component import ComponentType, create_component
from pytext.data.sources import PandasDataSource


class PandasDataSourceTest(unittest.TestCase):
    def test_create_from_config(self):
        source_config = PandasDataSource.Config(
            train_df=pd.DataFrame({"c1": [10, 20, 30], "c2": [40, 50, 60]}),
            eval_df=pd.DataFrame({"c1": [11, 21, 31], "c2": [41, 51, 61]}),
            test_df=pd.DataFrame({"c1": [12, 22, 32], "c2": [42, 52, 62]}),
            column_mapping={"c1": "feature1", "c2": "feature2"},
        )
        ds = create_component(
            ComponentType.DATA_SOURCE,
            source_config,
            schema={"feature1": float, "feature2": float},
        )
        self.assertEqual({"feature1": 10, "feature2": 40}, next(iter(ds.train)))
        self.assertEqual({"feature1": 11, "feature2": 41}, next(iter(ds.eval)))
        self.assertEqual({"feature1": 12, "feature2": 42}, next(iter(ds.test)))
        self.assertEqual(3, len(list(ds.train)))

    def test_create_data_source(self):
        ds = PandasDataSource(
            train_df=pd.DataFrame({"c1": [10, 20, 30], "c2": [40, 50, 60]}),
            eval_df=pd.DataFrame({"c1": [11, 21, 31], "c2": [41, 51, 61]}),
            test_df=pd.DataFrame({"c1": [12, 22, 32], "c2": [42, 52, 62]}),
            schema={"feature1": float, "feature2": float},
            column_mapping={"c1": "feature1", "c2": "feature2"},
        )
        self.assertEqual({"feature1": 10, "feature2": 40}, next(iter(ds.train)))
        self.assertEqual({"feature1": 11, "feature2": 41}, next(iter(ds.eval)))
        self.assertEqual({"feature1": 12, "feature2": 42}, next(iter(ds.test)))
        self.assertEqual(3, len(list(ds.train)))

    def test_empty_data(self):
        ds = PandasDataSource(schema={"feature1": float, "feature2": float})
        self.assertEqual(0, len(list(ds.train)))
