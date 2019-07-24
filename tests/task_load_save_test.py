#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import tempfile
import unittest

import torch
from pytext.config import LATEST_VERSION, PyTextConfig
from pytext.data import Data
from pytext.data.sources import TSVDataSource
from pytext.task import create_task
from pytext.task.serialize import load, save
from pytext.task.tasks import DocumentClassificationTask
from pytext.utils import test


tests_module = test.import_tests_module()


class TaskLoadSaveTest(unittest.TestCase):
    def assertModulesEqual(self, mod1, mod2, message=None):
        for p1, p2 in itertools.zip_longest(mod1.parameters(), mod2.parameters()):
            self.assertTrue(p1.equal(p2), message)

    def test_load_saved_model(self):
        with tempfile.NamedTemporaryFile() as snapshot_file:
            train_data = tests_module.test_file("train_data_tiny.tsv")
            eval_data = tests_module.test_file("test_data_tiny.tsv")
            config = PyTextConfig(
                task=DocumentClassificationTask.Config(
                    data=Data.Config(
                        source=TSVDataSource.Config(
                            train_filename=train_data,
                            eval_filename=eval_data,
                            field_names=["label", "slots", "text"],
                        )
                    )
                ),
                version=LATEST_VERSION,
                save_snapshot_path=snapshot_file.name,
            )
            task = create_task(config.task)
            model = task.model

            save(config, model, meta=None, tensorizers=task.data.tensorizers)
            task2, config2 = load(snapshot_file.name)

            self.assertEqual(config, config2)
            self.assertModulesEqual(model, task2.model)

            model.eval()
            task2.model.eval()

            inputs = torch.LongTensor([[1, 2, 3]]), torch.LongTensor([3])
            self.assertEqual(model(*inputs).tolist(), task2.model(*inputs).tolist())
