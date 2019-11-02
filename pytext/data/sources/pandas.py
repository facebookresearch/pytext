#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

from pandas import DataFrame
from pytext.data.sources.data_source import RootDataSource


class PandasDataSource(RootDataSource):
    """
    DataSource which loads data from a pandas DataFrame.

    Inputs:
        train_df: DataFrame for training

        eval_df: DataFrame for evalu

        test_df: DataFrame for test

        schema: same as base DataSource, define the list of output values with their types

        column_mapping: maps the column names in DataFrame to the name defined in schema

    """

    def __init__(
        self,
        train_df: Optional[DataFrame] = None,
        eval_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_df = train_df
        self.eval_df = eval_df
        self.test_df = test_df

    @staticmethod
    def raw_generator(df: Optional[DataFrame]):
        if df is None:
            yield from ()
        else:
            for _, row in df.iterrows():
                yield row

    def raw_train_data_generator(self):
        return self.raw_generator(self.train_df)

    def raw_eval_data_generator(self):
        return self.raw_generator(self.eval_df)

    def raw_test_data_generator(self):
        return self.raw_generator(self.test_df)
