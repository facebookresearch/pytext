#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import tempfile
import unittest

import numpy as np
from pytext import batch_predict_caffe2_model
from pytext.config import LATEST_VERSION, PyTextConfig
from pytext.data import Data
from pytext.data.sources import TSVDataSource
from pytext.data.tensorizers import (
    FloatListTensorizer,
    LabelTensorizer,
    TokenTensorizer,
)
from pytext.models.doc_model import DocModel
from pytext.task import create_task
from pytext.task.serialize import save
from pytext.task.tasks import DocumentClassificationTask
from pytext.utils import test


tests_module = test.import_tests_module()


class PredictorTest(unittest.TestCase):
    def test_batch_predict_caffe2_model(self):
        with tempfile.NamedTemporaryFile() as snapshot_file, tempfile.NamedTemporaryFile() as caffe2_model_file:
            train_data = tests_module.test_file("train_data_tiny.tsv")
            eval_data = tests_module.test_file("test_data_tiny.tsv")
            config = PyTextConfig(
                task=DocumentClassificationTask.Config(
                    model=DocModel.Config(
                        inputs=DocModel.Config.ModelInput(
                            tokens=TokenTensorizer.Config(),
                            dense=FloatListTensorizer.Config(
                                column="dense", dim=1, error_check=True
                            ),
                            labels=LabelTensorizer.Config(),
                        )
                    ),
                    data=Data.Config(
                        source=TSVDataSource.Config(
                            train_filename=train_data,
                            eval_filename=eval_data,
                            test_filename=eval_data,
                            field_names=["label", "slots", "text", "dense"],
                        )
                    ),
                ),
                version=LATEST_VERSION,
                save_snapshot_path=snapshot_file.name,
                export_caffe2_path=caffe2_model_file.name,
            )
            task = create_task(config.task)
            task.export(task.model, caffe2_model_file.name)
            model = task.model
            save(config, model, meta=None, tensorizers=task.data.tensorizers)

            results = batch_predict_caffe2_model(
                snapshot_file.name, caffe2_model_file.name
            )
            self.assertEqual(4, len(results))

            pt_results = task.predict(task.data.data_source.test)

            for pt_res, res in zip(pt_results, results):
                np.testing.assert_array_almost_equal(
                    pt_res["score"].tolist()[0], [score[0] for score in res.values()]
                )
