#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import random
from typing import List, Optional

from pytext.data.sources.data_source import DataSource, generator_property
from pytext.utils.file_io import PathManager


class DenseRetrievalDataSource(DataSource):
    """Data source for DPR (https://github.com/facebookresearch/DPR).

    Expects multiline json for lazy loading and improved memory usage.
    The original DPR files can be converted to multiline json using `jq -c .[]`
    """

    # TODO: Remove assumption that only 1 +ve passage is sample per question.
    DEFAULT_SCHEMA = {"question": str, "positive_ctx": str, "negative_ctxs": List[str]}

    class Config(DataSource.Config):
        train_filename: Optional[str] = "train-v2.0.json"
        test_filename: Optional[str] = "dev-v2.0.json"
        eval_filename: Optional[str] = "dev-v2.0.json"
        num_negative_ctxs: int = 1
        use_title: bool = True

    @classmethod
    def from_config(cls, config: Config, schema=DEFAULT_SCHEMA):
        return cls(
            schema=schema,
            train_filename=config.train_filename,
            test_filename=config.test_filename,
            eval_filename=config.eval_filename,
            num_negative_ctxs=config.num_negative_ctxs,
            use_title=config.use_title,
        )

    def __init__(
        self,
        schema,
        train_filename=None,
        test_filename=None,
        eval_filename=None,
        num_negative_ctxs=1,
        use_title=True,
    ):
        super().__init__(schema)
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.eval_filename = eval_filename
        self.num_negative_ctxs = num_negative_ctxs
        self.use_title = use_title

    @generator_property
    def train(self):
        return self.process_file(self.train_filename, is_train=True)

    @generator_property
    def test(self):
        return self.process_file(self.test_filename, is_train=False)

    @generator_property
    def eval(self):
        return self.process_file(self.eval_filename, is_train=False)

    def process_file(self, fname, is_train):
        if not fname:
            print(f"File path is either empty or None. Not unflattening.")
            return
        if not PathManager.exists(fname):
            print(f"{fname} does not exist. Not unflattening.")
            return

        with PathManager.open(fname) as infile:
            # Code pointer: https://fburl.com/yv8osgvo
            for line in infile:
                row = json.loads(line)
                question = row["question"]
                positive_ctx = combine_title_text(
                    row["positive_ctxs"][0], self.use_title
                )

                negative_ctxs = [
                    combine_title_text(ctx, self.use_title)
                    for ctx in row["negative_ctxs"]
                ]

                if not negative_ctxs and row.get("distant_negatives"):
                    # use distant_negatives in case we don't have hard negatives
                    # it's better to have at least one negative for training
                    negative_ctxs = [
                        combine_title_text(ctx, self.use_title)
                        for ctx in row["distant_negatives"]
                    ]

                if is_train:
                    random.shuffle(negative_ctxs)
                else:
                    # for non training runs, always take the num_negative_ctxs without shuffling
                    # this makes the evaluation and test sets deterministic
                    negative_ctxs = negative_ctxs[: self.num_negative_ctxs]

                num_negative_ctx = min(self.num_negative_ctxs, len(negative_ctxs))
                yield {
                    "question": question,
                    "positive_ctx": positive_ctx,
                    "negative_ctxs": negative_ctxs,
                    "label": "1",  # Make LabelTensorizer.initialize() happy.
                    "num_negative_ctx": num_negative_ctx,
                }


def combine_title_text(ctx, use_title):
    return (ctx["title"], ctx["text"]) if use_title else (ctx["text"],)
