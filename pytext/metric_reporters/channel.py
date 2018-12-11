#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import csv
import json
from typing import Tuple

from pytext.common.constants import Stage


class Channel:
    """
    Channel defines how to format and report the result of a PyText job to an output
    stream.

    Attributes:
        stages: in which stages the report will be triggered, default is all
            stages, which includes train, eval, test
    """

    def __init__(
        self, stages: Tuple[Stage, ...] = (Stage.TRAIN, Stage.EVAL, Stage.TEST)
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
        meta,
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
            meta (Dict[str, Any]): global metadata, such as target names
        """
        raise NotImplementedError()


class ConsoleChannel(Channel):
    """
    Simple Channel that prints results to console.
    """

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
        *args,
    ):
        print(f"\n\n{stage}")
        print(f"loss: {loss:.6f}")
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
        meta,
        *args,
    ):

        print(f"saving result to file {self.file_path}")
        with open(self.file_path, "w", encoding="utf-8") as of:
            for metadata in meta.values():
                # TODO the # prefix is quite ad-hoc, we should think of a better
                # way to handle it
                of.write("#")
                of.write(json.dumps(metadata))
                of.write("\n")

            tsv_writer = csv.writer(
                of,
                delimiter="\t",
                quotechar='"',
                doublequote=True,
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )

            tsv_writer.writerow(self.get_title())
            for row in self.gen_content(
                metrics, loss, preds, targets, scores, context, *args
            ):
                tsv_writer.writerow(row)

    def get_title(self):
        return ("prediction", "target", "score")

    def gen_content(self, metrics, loss, preds, targets, scores, contexts):
        for i in range(len(preds)):
            yield [preds[i], targets[i], scores[i]]


class TensorBoardChannel(Channel):
    """
    TensorBoardChannel defines how to format and report the result of a PyText
    job to TensorBoard.

    Attributes:
        summary_writer: An instance of the TensorBoardX SummaryWriter class, or
            an object that implements the same interface.
            https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
        metric_name: The name of the default metric to display on the
            TensorBoard dashboard, defaults to "accuracy"
    """

    def __init__(self, summary_writer, metric_name="accuracy"):
        super().__init__()
        self.summary_writer = summary_writer
        self.metric_name = metric_name

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
        *args,
    ):
        """
        Defines how to format and report data to TensorBoard using the summary
        writer. In the current implementation, during the train/eval phase we
        recursively report each metric field as scalars, and during the test
        phase we report the final metrics to be displayed as texts.
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
        """
        if stage == Stage.TEST:
            tag = "test"
            self.summary_writer.add_text(tag, f"loss={loss}")
            if isinstance(metrics, (int, float)):
                self.summary_writer.add_text(tag, f"{self.metric_name}={metrics}")
            else:
                self.add_texts(tag, metrics)
        else:
            prefix = "train" if stage == Stage.TRAIN else "eval"
            self.summary_writer.add_scalar(f"{prefix}/loss", loss, epoch)
            if isinstance(metrics, (int, float)):
                self.summary_writer.add_scalar(
                    f"{prefix}/{self.metric_name}", metrics, epoch
                )
            else:
                self.add_scalars(prefix, metrics, epoch)

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
        for field_name, field_value in metrics._asdict().items():
            if isinstance(field_value, (int, float)):
                self.summary_writer.add_scalar(
                    f"{prefix}/{field_name}", field_value, epoch
                )
            elif hasattr(field_value, "_asdict"):
                self.add_scalars(f"{prefix}/{field_name}", field_value, epoch)
