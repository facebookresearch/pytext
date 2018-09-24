#!/usr/bin/env python3


import csv
import json
from typing import Tuple

from pytext.common.constants import Stage


class Channel:
    """Channel defines a way to report the result of a PyText job
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
    ):
        raise NotImplementedError()


class ConsoleChannel(Channel):
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
    ):
        print(f"{stage}")
        print(f"loss: {loss:.6f}")
        # TODO change print_metrics function to __str__ T33522209
        if hasattr(metrics, "print_metrics"):
            metrics.print_metrics()
        else:
            print(metrics)


class FileChannel(Channel):
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
            for row in self.gen_content(metrics, loss, preds, targets, scores, context):
                tsv_writer.writerow(row)

    def get_title(self):
        return ("prediction", "target", "score")

    def gen_content(self, metrics, loss, preds, targets, scores, contexts):
        for i in range(len(preds)):
            yield [preds[i], targets[i], scores[i]]
