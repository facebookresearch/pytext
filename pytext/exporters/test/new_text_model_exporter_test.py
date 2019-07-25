#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import tempfile
from typing import Type

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.predictor.predictor_exporter as pe
import numpy as np
import torch.nn.functional as F
from caffe2.python import workspace
from pytext.common.constants import Stage
from pytext.config import LATEST_VERSION, PyTextConfig
from pytext.data import Data, PoolingBatcher
from pytext.data.sources import TSVDataSource
from pytext.data.tensorizers import FloatListTensorizer
from pytext.models.doc_model import DocModel, PersonalizedDocModel
from pytext.models.model import Model
from pytext.task.new_task import NewTask
from pytext.task.tasks import DocumentClassificationTask
from pytext.trainers import TaskTrainer
from pytext.utils.onnx import CAFFE2_DB_TYPE
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()
TestFileMetadata = tests_module.TestFileMetadata
TestFileName = tests_module.TestFileName
get_test_file_metadata = tests_module.get_test_file_metadata

test_file_and_models = [
    (TestFileName.TRAIN_DENSE_FEATURES_TINY_TSV, DocModel),
    (TestFileName.TEST_PERSONALIZATION_OPPOSITE_INPUTS_TSV, PersonalizedDocModel),
]


class ModelExporterTest(hu.HypothesisTestCase):
    @staticmethod
    def _feed_c2_input(workspace, py_inputs, input_names, vocab_map):
        c2_input = []
        for py_input in py_inputs:
            c2_input = c2_input + (
                list(py_input) if isinstance(py_input, tuple) else [py_input]
            )
        for i, input in enumerate(list(c2_input)):
            input_np = input.numpy()
            if input_names[i] in vocab_map.keys():
                # Map the input to the str form
                input_vocab = vocab_map[input_names[i]]
                map_fn = np.vectorize(lambda x: input_vocab[x])
                input_str = map_fn(input_np)
                input_np = np.array(input_str, dtype=str)
                workspace.FeedBlob(input_names[i] + "_str:value", input_np)
            else:
                workspace.FeedBlob(input_names[i], input_np)

    def _get_config(
        self,
        task_class: Type[NewTask],
        model_class: Type[Model],
        test_file_metadata: TestFileMetadata,
    ) -> PyTextConfig:
        return PyTextConfig(
            task=task_class.Config(
                data=Data.Config(
                    source=TSVDataSource.Config(
                        train_filename=test_file_metadata.filename,
                        eval_filename=test_file_metadata.filename,
                        test_filename=test_file_metadata.filename,
                        field_names=test_file_metadata.field_names,
                    ),
                    batcher=PoolingBatcher.Config(
                        train_batch_size=1, test_batch_size=1
                    ),
                ),
                trainer=TaskTrainer.Config(epochs=1),
                model=model_class.Config(
                    inputs=type(model_class.Config.inputs)(
                        dense=FloatListTensorizer.Config(
                            column=test_file_metadata.dense_col_name,
                            error_check=True,
                            dim=test_file_metadata.dense_feat_dim,
                        )
                    )
                ),
            ),
            use_tensorboard=False,
            use_cuda_if_available=False,
            export_torchscript_path="/tmp/model_torchscript.pt",
            version=LATEST_VERSION,
        )

    def _test_task_export_to_caffe2(self, task_class, config):
        task = task_class.from_config(config.task)
        py_model = task.model
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format("predictor")
        ) as pred_file:
            print(pred_file.name)
            output_names = task.export(py_model, pred_file.name)
            print(output_names)
            workspace.ResetWorkspace()

        pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)

        test_inputs = py_model.arrange_model_inputs(
            next(iter(task.data.batches(Stage.TEST)))[1]
        )
        ModelExporterTest._feed_c2_input(
            workspace,
            test_inputs,
            # get_export_input_names only implemented for document classification
            task.model.get_export_input_names(task.data.tensorizers),
            task.model.vocab_to_export(task.data.tensorizers),
        )
        workspace.RunNetOnce(pred_net)
        c2_out = [list(workspace.FetchBlob(o_name)) for o_name in output_names]

        py_model.eval()
        py_outs = py_model(*test_inputs)
        # Do softmax since we do log softmax before exporting predictor nets
        # We do exp on caffe2 output instead, because log of numbers that are
        # very close to 0 gives different result in pytorch and caffe2
        py_outs = F.softmax(py_outs, 1)

        np.testing.assert_array_almost_equal(
            py_outs.view(-1).detach().numpy(),
            np.exp(np.array(c2_out).transpose()).flatten(),
        )

    def test_document_export_to_caffe2(self):
        task_class = DocumentClassificationTask

        for test_file_name, model_class in test_file_and_models:
            config = self._get_config(
                task_class=task_class,
                model_class=model_class,
                test_file_metadata=get_test_file_metadata(test_file_name),
            )
            self._test_task_export_to_caffe2(task_class=task_class, config=config)
