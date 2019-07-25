#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import unittest
from typing import Tuple, Type

import torch
from pytext.config import LATEST_VERSION, PyTextConfig
from pytext.data import Batcher, Data
from pytext.data.sources.tsv import TSVDataSource
from pytext.data.tensorizers import FloatListTensorizer
from pytext.models.doc_model import DocModel, PersonalizedDocModel
from pytext.models.model import Model
from pytext.optimizer.optimizers import SGD
from pytext.task.new_task import NewTask
from pytext.task.tasks import DocumentClassificationTask
from pytext.trainers import TaskTrainer
from pytext.utils.model import get_mismatched_param
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()
TestFileName = tests_module.TestFileName
get_test_file_metadata = tests_module.get_test_file_metadata


class PersonalizationTrainerTest(unittest.TestCase):
    def _get_pytext_config(
        self,
        test_file_name: TestFileName,
        task_class: Type[NewTask],
        model_class: Type[Model],
    ) -> PyTextConfig:
        test_file_metadata = get_test_file_metadata(test_file_name)
        return PyTextConfig(
            task=task_class.Config(
                data=Data.Config(
                    source=TSVDataSource.Config(
                        train_filename=test_file_metadata.filename,
                        eval_filename=test_file_metadata.filename,
                        test_filename=test_file_metadata.filename,
                        field_names=test_file_metadata.field_names,
                    ),
                    batcher=Batcher.Config(),  # Use Batcher to avoid shuffling.
                ),
                trainer=TaskTrainer.Config(epochs=1),
                model=model_class.Config(
                    inputs=type(model_class.Config.inputs)(
                        dense=FloatListTensorizer.Config(
                            column=test_file_metadata.dense_col_name,
                            dim=test_file_metadata.dense_feat_dim,
                        )
                    )
                ),
            ),
            use_tensorboard=False,
            use_cuda_if_available=False,
            version=LATEST_VERSION,
        )

    def _get_task(
        self,
        test_file_name: TestFileName,
        task_class: Type[NewTask],
        model_class: Type[Model],
    ) -> Tuple[PyTextConfig, NewTask]:
        pytext_config = self._get_pytext_config(test_file_name, task_class, model_class)
        task = task_class.from_config(pytext_config.task)
        return pytext_config, task

    def _get_baseline_task(
        self, test_file_name: TestFileName
    ) -> Tuple[PyTextConfig, NewTask]:
        return self._get_task(
            test_file_name=test_file_name,
            task_class=DocumentClassificationTask,
            model_class=DocModel,
        )

    def _get_p13n_task(
        self, test_file_name: TestFileName
    ) -> Tuple[PyTextConfig, NewTask]:
        return self._get_task(
            test_file_name=test_file_name,
            task_class=DocumentClassificationTask,
            model_class=PersonalizedDocModel,
        )

    def test_p13n_performance(self):
        test_file_name = TestFileName.TEST_PERSONALIZATION_OPPOSITE_INPUTS_TSV
        baseline_config, baseline_task = self._get_baseline_task(
            test_file_name=test_file_name
        )
        p13n_config, p13n_task = self._get_p13n_task(test_file_name=test_file_name)

        orig_p13n_model = copy.deepcopy(p13n_task.model)

        baseline_model, baseline_metrics = baseline_task.train(baseline_config)
        p13n_model, p13n_metrics = p13n_task.train(p13n_config)

        # Verify that the training changes the p13n_model.
        self.assertNotEqual(get_mismatched_param([orig_p13n_model, p13n_model]), "")

        # The dataset has two users with opposite inputs so a personalized model
        # is expected to have better performance.
        epsilon = 0.15
        self.assertGreater(p13n_metrics.roc_auc - baseline_metrics.roc_auc, epsilon)

    def test_user_embedding_updates(self):
        """Verify the user embeddings learn independently."""
        task_class = DocumentClassificationTask
        pytext_config = self._get_pytext_config(
            test_file_name=TestFileName.TEST_PERSONALIZATION_SINGLE_USER_TSV,
            task_class=task_class,
            model_class=PersonalizedDocModel,
        )
        # SGD changes only the user embeddings which have non-zero gradients.
        pytext_config.task.trainer.optimizer = SGD.Config()
        p13n_task = task_class.from_config(pytext_config.task)

        orig_user_embedding_weights = copy.deepcopy(
            p13n_task.model.user_embedding.weight
        )
        p13n_model, _ = p13n_task.train(pytext_config)
        trained_user_embedding_weights = p13n_model.user_embedding.weight

        self.assertEqual(
            len(orig_user_embedding_weights),
            2,
            "There should be 2 user embeddings, including the unknown user.",
        )

        self.assertEqual(
            len(orig_user_embedding_weights),
            len(trained_user_embedding_weights),
            "Length of user embeddings should not be changed by the training.",
        )

        # Verify that the training changes only 1 user embedding in the p13n_model.
        self.assertTrue(
            torch.equal(
                orig_user_embedding_weights[0], trained_user_embedding_weights[0]
            ),
            "Unknown user embedding should not change.",
        )
        self.assertFalse(
            torch.equal(
                orig_user_embedding_weights[1], trained_user_embedding_weights[1]
            ),
            "The only user embedding should change.",
        )
