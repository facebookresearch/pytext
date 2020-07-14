#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional, Type

from pandas import DataFrame
from pytext.data.sources.data_source import RootDataSource

from .session import SessionDataSource


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

    class Config(RootDataSource.Config):
        train_df: Optional[DataFrame] = None
        test_df: Optional[DataFrame] = None
        eval_df: Optional[DataFrame] = None

    @classmethod
    def from_config(cls, config: Config, schema: Dict[str, Type]):
        return cls(
            train_df=config.train_df,
            eval_df=config.eval_df,
            test_df=config.test_df,
            schema=schema,
            column_mapping=config.column_mapping,
        )

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


class SessionPandasDataSource(PandasDataSource, SessionDataSource):
    def __init__(
        self,
        schema: Dict[str, Type],
        id_col: str,
        train_df: Optional[DataFrame] = None,
        eval_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        column_mapping: Dict[str, str] = (),
    ):
        schema[id_col] = str
        super().__init__(
            schema=schema,
            train_df=train_df,
            test_df=test_df,
            eval_df=eval_df,
            column_mapping=column_mapping,
            id_col=id_col,
        )
        self._validate_schema()
