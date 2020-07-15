#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import csv
import sys
import traceback
from typing import Tuple

import numpy as np
import torch
from numpy import linalg as LA
from pytext.common.constants import Stage
from pytext.utils.file_io import PathManager
from torch.utils.tensorboard import SummaryWriter


try:
    from torch.utils.tensorboard.fb.profile import profile_graph
except ImportError:
    profile_graph = None


class Channel:
    """
    Channel defines how to format and report the result of a PyText job to an output
    stream.

    Attributes:
        stages: in which stages the report will be triggered, default is all
            stages, which includes train, eval, test
    """

    def __init__(
        self,
        stages: Tuple[Stage, ...] = (Stage.TRAIN, Stage.EVAL, Stage.TEST, Stage.OTHERS),
    ) -> None:
        self.stages = stages

    def report(
        self,
        stage,
        epoch,
        metrics,
        model_select_metric,
        loss,
        preds,
        targets,
        scores,
        context,
        *args,
    ):
        """
        Defines how to format and report data to the output channel.

        Args:
            stage (Stage): train, eval or test
            epoch (int): current epoch
            metrics (Any): all metrics
            model_select_metric (double): a single numeric metric to pick best model
            loss (double): average loss
            preds (List[Any]): list of predictions
            targets (List[Any]): list of targets
            scores (List[Any]): list of scores
            context (Dict[str, List[Any]]): dict of any additional context data,
                each context is a list of data that maps to each example
        """
        raise NotImplementedError()

    def close(self):
        pass

    def export(self, model, input_to_model=None, **kwargs):
        pass


class ConsoleChannel(Channel):
    """
    Simple Channel that prints results to console.
    """

    def _print_loss(self, loss):
        if isinstance(loss, float):
            print(f"loss: {loss:.6f}")
        elif isinstance(loss, dict):
            for key in loss:
                print(f"{key}: {loss[key]:.6f}")
        else:
            raise Exception("Loss type not supported")

    def report(
        self,
        stage,
        epoch,
        metrics,
        model_select_metric,
        loss,
        preds,
        targets,
        scores,
        context,
        *args,
    ):
        print(f"\n\n{stage}")
        print(f"Epoch:{epoch}")
        self._print_loss(loss)
        # TODO change print_metrics function to __str__ T33522209
        if hasattr(metrics, "print_metrics"):
            metrics.print_metrics()
        else:
            print(metrics)


class FileChannel(Channel):
    """
    Simple Channel that writes results to a TSV file.
    """

    def __init__(self, stages, file_path) -> None:
        super().__init__(stages)
        self.file_path = file_path

    def report(
        self,
        stage,
        epoch,
        metrics,
        model_select_metric,
        loss,
        preds,
        targets,
        scores,
        context,
        *args,
    ):
        print(f"saving result to file {self.file_path}")
        with PathManager.open(self.file_path, "w", encoding="utf-8") as of:
            tsv_writer = csv.writer(
                of,
                delimiter="\t",
                quotechar='"',
                doublequote=True,
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )

            tsv_writer.writerow(self.get_title(tuple(context.keys())))
            for row in self.gen_content(metrics, loss, preds, targets, scores, context):
                tsv_writer.writerow(row)

    def get_title(self, context_keys=()):
        return ("prediction", "target", "score") + context_keys

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        context_values = context.values()
        for i in range(len(preds)):
            # if we are running the metric reporter in memory_efficient mode
            # then we don't store any scores
            if len(scores) == 0:
                res = [preds[i], targets[i]]
            else:
                res = [preds[i], targets[i], scores[i]]
            res.extend([v_list[i] for v_list in context_values])
            yield res


class TensorBoardChannel(Channel):
    """
    TensorBoardChannel defines how to format and report the result of a PyText
    job to TensorBoard.

    Attributes:
        summary_writer: An instance of the TensorBoard SummaryWriter class, or
            an object that implements the same interface.
            https://pytorch.org/docs/stable/tensorboard.html
        metric_name: The name of the default metric to display on the
            TensorBoard dashboard, defaults to "accuracy"
        train_step: The training step count
    """

    def __init__(self, summary_writer=None, metric_name="accuracy"):
        super().__init__()
        self.summary_writer = summary_writer or SummaryWriter()
        self.metric_name = metric_name

    def log_loss(self, prefix, loss, epoch):
        if isinstance(loss, float):
            self.summary_writer.add_scalar(f"{prefix}/loss", loss, epoch)
        elif isinstance(loss, dict):
            for key in loss:
                self.summary_writer.add_scalar(f"{prefix}/" + key, loss[key], epoch)
        else:
            raise Exception("Loss type not supported")

    def report(
        self,
        stage,
        epoch,
        metrics,
        model_select_metric,
        loss,
        preds,
        targets,
        scores,
        context,
        meta,
        model,
        optimizer,
        log_gradient,
        gradients,
        *args,
    ):
        """
        Defines how to format and report data to TensorBoard using the summary
        writer. In the current implementation, during the train/eval phase we
        recursively report each metric field as scalars, and during the test
        phase we report the final metrics to be displayed as texts.

        Also visualizes the internal model states (weights, biases) as
        histograms in TensorBoard.

        Args:
            stage (Stage): train, eval or test
            epoch (int): current epoch
            metrics (Any): all metrics
            model_select_metric (double): a single numeric metric to pick best model
            loss (double): average loss
            preds (List[Any]): list of predictions
            targets (List[Any]): list of targets
            scores (List[Any]): list of scores
            context (Dict[str, List[Any]]): dict of any additional context data,
                each context is a list of data that maps to each example
            meta (Dict[str, Any]): global metadata, such as target names
            model (nn.Module): the PyTorch neural network model
        """

        def stage2prefix(stage: Stage):
            """
            mapping a Stage to a specific tag for printing into TB. Sometimes
            we may have a subclass of Stage that includes additional stages for
            fine-grained bookkeeping, which is mapped to "other" in TB.
            """
            if stage == Stage.TRAIN:
                return "train"
            elif stage == Stage.EVAL:
                return "eval"
            elif stage == Stage.TEST:
                return "test"
            else:
                return "others"

        if stage == Stage.TEST:
            tag = "test"
            self.summary_writer.add_text(tag, f"loss={loss}")
            if isinstance(metrics, (int, float)):
                self.summary_writer.add_text(tag, f"{self.metric_name}={metrics}")
            else:
                self.add_texts(tag, metrics)
        else:
            prefix = stage2prefix(stage)
            self.log_loss(prefix, loss, epoch)
            if isinstance(metrics, (int, float)):
                self.summary_writer.add_scalar(
                    f"{prefix}/{self.metric_name}", metrics, epoch
                )
            else:
                self.add_scalars(prefix, metrics, epoch)

        if stage == Stage.TRAIN:
            if optimizer is not None:
                for idx, param_group in enumerate(optimizer.param_groups):
                    self.summary_writer.add_scalar(
                        f"optimizer.lr.param_group.{idx}", param_group["lr"], epoch
                    )
            if log_gradient and gradients:
                for key in gradients:
                    if len(gradients[key]):
                        sum_gradient = sum(gradients[key])
                        avg_gradient = sum_gradient / len(gradients[key])
                        grad_norms = np.array([LA.norm(g) for g in gradients[key]])
                        self.log_vector(key + "_avg_gradients", avg_gradient, epoch)
                        self.log_vector(key + "_sum_gradients", sum_gradient, epoch)
                        self.log_vector(key + "_l2norm_gradients", grad_norms, epoch)

            for key, val in model.named_parameters():
                if val is not None and len(val) > 0 and not (val == 0).all():
                    limit = 9.9e19
                    val = torch.clamp(val.float(), -limit, limit)
                    self.log_vector(key, val, epoch)

    def log_vector(self, key, val, epoch):
        if len(val) > 0 and not (val == 0).all():
            try:
                self.summary_writer.add_histogram(key, val, epoch)
            except Exception:
                print(
                    f"WARNING: Param {key} " "cannot be sent to Tensorboard",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)

    def add_texts(self, tag, metrics):
        """
        Recursively flattens the metrics object and adds each field name and
        value as a text using the summary writer. For example, if tag = "test",
        and metrics = { accuracy: 0.7, scores: { precision: 0.8, recall: 0.6 } },
        then under "tag=test" we will display "accuracy=0.7", and under
        "tag=test/scores" we will display "precision=0.8" and "recall=0.6" in
        TensorBoard.

        Args:
            tag (str): The tag name for the metric. If a field needs to be
                flattened further, it will be prepended as a prefix to the field
                name.
            metrics (Any): The metrics object.
        """
        for field_name, field_value in metrics._asdict().items():
            if isinstance(field_value, (int, float)):
                self.summary_writer.add_text(tag, f"{field_name}={field_value}")
            elif hasattr(field_value, "_asdict"):
                self.add_texts(f"{tag}/{field_name}", field_value)

    def add_scalars(self, prefix, metrics, epoch):
        """
        Recursively flattens the metrics object and adds each field name and
        value as a scalar for the corresponding epoch using the summary writer.

        Args:
            prefix (str): The tag prefix for the metric. Each field name in the
                metrics object will be prepended with the prefix.
            metrics (Any): The metrics object.
        """
        if hasattr(metrics, "_asdict"):
            metrics = metrics._asdict()
        for field_name, field_value in metrics.items():
            if isinstance(field_value, (int, float)):
                self.summary_writer.add_scalar(
                    f"{prefix}/{field_name}", field_value, epoch
                )
            elif hasattr(field_value, "_asdict") or isinstance(field_value, dict):
                self.add_scalars(f"{prefix}/{field_name}", field_value, epoch)

    def close(self):
        """
        Closes the summary writer.
        """
        self.summary_writer.close()

    def export(self, model, input_to_model=None, **kwargs):
        """
        Draws the neural network representation graph in TensorBoard.

        Args:
            model (Any): the model object.
            input_to_model (Any): the input to the model (required for PyTorch
                models, since its execution graph is defined by run).
        """
        try:
            self.summary_writer.add_graph(model, input_to_model, **kwargs)
        except Exception:
            print(
                "WARNING: Unable to export neural network graph to TensorBoard.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

        try:
            if profile_graph is not None:
                profile_graph(self.summary_writer, model, input_to_model)
        except Exception:
            print(
                "WARNING: Unable to export performance graph to Tensor Board.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

        try:
            model.embedding.visualize(self.summary_writer)
        except Exception:
            print(
                "WARNING: Unable to visualize embedding space in TensorBoard.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
