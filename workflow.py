#!/usr/bin/env python3

import csv
import os

import torch
from pytext.config.component import (
    create_data_handler,
    create_exporter,
    create_loss,
    create_model,
    create_trainer,
)
from pytext.optimizer import create_optimizer, create_scheduler
from pytext.config import PyTextConfig

from .serialize import load, save
from .utils import cuda_utils


def _set_cuda(use_cuda_if_available):
    cuda_utils.CUDA_ENABLED = use_cuda_if_available and torch.cuda.is_available()
    if use_cuda_if_available and not cuda_utils.CUDA_ENABLED:
        print("Cuda is not available, running on CPU...")


def train_model(config: PyTextConfig, metrics_reporter=None):
    print("\nParameters:\n{}".format(config))
    _set_cuda(config.use_cuda_if_available)
    jobspec = config.jobspec
    # load data
    data_handler = create_data_handler(
        jobspec.data_handler, jobspec.features, jobspec.labels
    )

    print("\nLoading data...")
    data_handler.init_metadata_from_path(
        config.train_path, config.eval_path, config.test_path
    )

    train_iter, eval_iter = data_handler.get_train_batch_from_path(
        (config.train_path, config.eval_path),
        (config.train_batch_size, config.eval_batch_size),
    )
    # load or create model
    metadata = data_handler.metadata
    if config.load_snapshot_path is None or not os.path.isfile(
        config.load_snapshot_path
    ):
        model = create_model(jobspec.model, jobspec.features, metadata)
    else:
        print("\nLoading model from [%s]..." % config.load_snapshot_path)
        model = load(config.load_snapshot_path)["model"]

    if cuda_utils.CUDA_ENABLED:
        model = model.cuda()

    optimizers = create_optimizer(model, jobspec.optimizer)
    scheduler = create_scheduler(optimizers, jobspec.scheduler)
    trainer = create_trainer(jobspec.trainer, metadata)

    trained_model = trainer.train(
        train_iter,
        eval_iter,
        model,
        optimizers,
        metadata.labels,
        metrics_reporter,
        scheduler
    )

    print("Saving pytorch model to: " + config.save_snapshot_path)
    # Make sure to put the model on CPU and disable CUDA before exporting to
    # ONNX to disable any data_parallel pieces
    _set_cuda(False)
    save(config.save_snapshot_path, config, trained_model.cpu(), data_handler)

    if config.jobspec.exporter:
        print("Saving caffe2 model to: " + config.export_caffe2_path)
        exporter = create_exporter(
            jobspec.exporter, jobspec.features, jobspec.labels, data_handler.metadata
        )
        exporter.export_to_caffe2(trained_model, config.export_caffe2_path)


def test_model(config):
    _set_cuda(config.use_cuda_if_available)
    model_path = config.load_snapshot_path
    if model_path is None or not os.path.isfile(model_path):
        raise ValueError("Invalid snapshot path for testing: {}".format(model_path))

    print("\nLoading model from [%s]..." % model_path)

    train_config, model, data_handler = load(model_path)
    model.eval()

    if cuda_utils.CUDA_ENABLED:
        model = model.cuda()

    # TODO T31914569 should move test out of trainer?
    trainer = create_trainer(config.jobspec.trainer, data_handler.metadata)
    test_iter = data_handler.get_test_batch(
        config.test_path, config.test_batch_size
    )
    results, metrics = trainer.test(model, test_iter, data_handler.metadata)

    with open(config.test_out_path, "w+", encoding="utf-8") as of:
        tsv_writer = csv.writer(
            of,
            delimiter="\t",
            quotechar='"',
            doublequote=True,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        for l in results:
            if l[0] == "#":
                of.write(l)
                of.write("\n")
            else:
                tsv_writer.writerow([f for f in l])

    return metrics
