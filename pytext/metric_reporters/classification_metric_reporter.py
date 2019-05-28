#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import List, Optional

from pytext.common.constants import Stage
from pytext.data import CommonMetadata
from pytext.metrics import LabelPrediction, compute_classification_metrics

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


META_LABEL_NAMES = "label_names"


class IntentModelChannel(FileChannel):
    def get_title(self):
        return ("predicted", "actual", "scores_str", "text")

    def gen_content(self, metrics, loss, preds, targets, scores, contexts):
        for i in range(len(preds)):
            yield [
                preds[i],
                targets[i],
                ",".join([f"{s:.2f}" for s in scores[i]]),
                contexts["utterance"][i],
            ]


class ComparableClassificationMetric(Enum):
    ACCURACY = "accuracy"
    ROC_AUC = "roc_auc"
    MCC = "mcc"
    MACRO_F1 = "macro_f1"
    LABEL_F1 = "label_f1"
    LABEL_AVG_PRECISION = "label_avg_precision"
    LABEL_ROC_AUC = "label_roc_auc"
    # use negative because the reporter's lower_is_better value is False
    NEGATIVE_LOSS = "negative_loss"


class ClassificationMetricReporter(MetricReporter):
    class Config(MetricReporter.Config):
        model_select_metric: ComparableClassificationMetric = (
            ComparableClassificationMetric.ACCURACY
        )
        target_label: Optional[str] = None
        #: These column names correspond to raw input data columns. Text in these
        #: columns (usually just 1 column) will be concatenated and output in
        #: the IntentModelChannel as an evaluation tsv.
        text_column_names: List[str] = ["text"]

    def __init__(
        self,
        label_names: List[str],
        channels: List[Channel],
        model_select_metric: ComparableClassificationMetric = (
            ComparableClassificationMetric.ACCURACY
        ),
        target_label: Optional[str] = None,
        text_column_names: List[str] = Config.text_column_names,
    ) -> None:
        super().__init__(channels)
        self.label_names = label_names
        self.model_select_metric = model_select_metric
        self.target_label = target_label
        self.text_column_names = text_column_names

    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        # TODO: refactor metric reporting and remove this hack
        if tensorizers:
            labels = list(tensorizers["labels"].vocab)
        else:
            labels = meta.target.vocab.itos
        return cls.from_config_and_label_names(config, labels)

    @classmethod
    def from_config_and_label_names(cls, config, label_names: List[str]):
        if config.model_select_metric in (
            ComparableClassificationMetric.LABEL_F1,
            ComparableClassificationMetric.LABEL_AVG_PRECISION,
            ComparableClassificationMetric.LABEL_ROC_AUC,
        ):
            assert config.target_label is not None
            assert config.target_label in label_names
        if config.model_select_metric in (
            ComparableClassificationMetric.ROC_AUC,
            ComparableClassificationMetric.MCC,
        ):
            assert len(label_names) == 2

        return cls(
            label_names,
            [ConsoleChannel(), IntentModelChannel((Stage.TEST,), config.output_path)],
            config.model_select_metric,
            config.target_label,
            config.text_column_names,
        )

    def batch_context(self, raw_batch, batch):
        context = super().batch_context(raw_batch, batch)
        context["utterance"] = [
            " | ".join(str(row[column_name]) for column_name in self.text_column_names)
            for row in raw_batch
        ]
        return context

    def calculate_metric(self):
        return compute_classification_metrics(
            [
                LabelPrediction(scores, pred, expect)
                for scores, pred, expect in zip(
                    self.all_scores, self.all_preds, self.all_targets
                )
            ],
            self.label_names,
            self.calculate_loss(),
        )

    def get_meta(self):
        return {META_LABEL_NAMES: self.label_names}

    def get_model_select_metric(self, metrics):
        if self.model_select_metric == ComparableClassificationMetric.ACCURACY:
            metric = metrics.accuracy
        elif self.model_select_metric == ComparableClassificationMetric.ROC_AUC:
            metric = metrics.roc_auc
        elif self.model_select_metric == ComparableClassificationMetric.MCC:
            metric = metrics.mcc
        elif self.model_select_metric == ComparableClassificationMetric.MACRO_F1:
            metric = metrics.macro_prf1_metrics.macro_scores.f1
        elif self.model_select_metric == ComparableClassificationMetric.LABEL_F1:
            metric = metrics.macro_prf1_metrics.per_label_scores[self.target_label].f1
        elif (
            self.model_select_metric
            == ComparableClassificationMetric.LABEL_AVG_PRECISION
        ):
            metric = metrics.per_label_soft_scores[self.target_label].average_precision
        elif self.model_select_metric == ComparableClassificationMetric.LABEL_ROC_AUC:
            metric = metrics.per_label_soft_scores[self.target_label].roc_auc
        elif self.model_select_metric == ComparableClassificationMetric.NEGATIVE_LOSS:
            metric = -metrics.loss
        else:
            raise ValueError(f"unknown metric: {self.model_select_metric}")

        assert metric is not None
        return metric
