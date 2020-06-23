#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from itertools import tee, zip_longest
from typing import Generator, List, Optional

import torch
from pytext.common.constants import Stage
from pytext.data import CommonMetadata
from pytext.metrics import (
    RECALL_AT_PRECISION_THRESHOLDS,
    LabelListPrediction,
    LabelPrediction,
    compute_classification_metrics,
    compute_multi_label_classification_metrics,
)

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


META_LABEL_NAMES = "label_names"


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
    __EXPANSIBLE__ = True

    class Config(MetricReporter.Config):
        model_select_metric: ComparableClassificationMetric = (
            ComparableClassificationMetric.ACCURACY
        )
        target_label: Optional[str] = None
        #: These column names correspond to raw input data columns. Text in these
        #: columns (usually just 1 column) will be concatenated and output in
        #: the IntentModelChannel as an evaluation tsv.
        text_column_names: List[str] = ["text"]
        #: These column names correspond to raw input data columns, that
        #: will be read by data_source into context, and included in the
        #: run_model output file along with other saving results.
        additional_column_names: List[str] = []
        recall_at_precision_thresholds: List[float] = RECALL_AT_PRECISION_THRESHOLDS
        # Boolean which is used to run a more memory efficient version of the
        # metric reporter. This involves storing as little information as possible
        # in memory and as a result, we don't compute metrics other than accuracy
        # and F1. This is useful when the label space is huge and we don't want to
        # keep around the score for every label for every example in memory.
        # This also means that the output debug file in the Test operator will have
        # only predictions and no label scores.
        is_memory_efficient: bool = False

    def __init__(
        self,
        label_names: List[str],
        channels: List[Channel],
        model_select_metric: ComparableClassificationMetric = (
            ComparableClassificationMetric.ACCURACY
        ),
        target_label: Optional[str] = None,
        text_column_names: List[str] = Config.text_column_names,
        additional_column_names: List[str] = Config.additional_column_names,
        recall_at_precision_thresholds: List[float] = (
            Config.recall_at_precision_thresholds
        ),
        is_memory_efficient: bool = Config.is_memory_efficient,
    ) -> None:
        super().__init__(channels)
        self.label_names = label_names
        self.model_select_metric = model_select_metric
        self.target_label = target_label
        self.text_column_names = text_column_names
        self.additional_column_names = additional_column_names
        self.recall_at_precision_thresholds = recall_at_precision_thresholds
        self.is_memory_efficient = is_memory_efficient

    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        # TODO: refactor metric reporting and remove this hack
        if tensorizers:
            labels = list(tensorizers["labels"].vocab)
        else:
            labels = meta.target.vocab.itos
            config.text_column_names = []
        return cls.from_config_and_label_names(config, labels)

    @classmethod
    def from_config_and_label_names(cls, config, label_names: List[str]):
        if config.model_select_metric in (
            ComparableClassificationMetric.LABEL_F1,
            ComparableClassificationMetric.LABEL_AVG_PRECISION,
            ComparableClassificationMetric.LABEL_ROC_AUC,
        ):
            assert (
                config.target_label is not None
            ), "target_label must be set for selected metric"
            assert config.target_label in label_names
        if config.model_select_metric in (
            ComparableClassificationMetric.ROC_AUC,
            ComparableClassificationMetric.MCC,
        ):
            assert (
                len(label_names) == 2
            ), "selected metric is valid for binary labels only"

        return cls(
            label_names,
            [ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
            config.model_select_metric,
            config.target_label,
            config.text_column_names,
            config.additional_column_names,
            config.recall_at_precision_thresholds,
            config.is_memory_efficient,
        )

    def batch_context(self, raw_batch, batch):
        context = super().batch_context(raw_batch, batch)
        context["text"] = [
            " | ".join(str(row[column_name]) for column_name in self.text_column_names)
            for row in raw_batch
        ]
        # if there are additional colnames, read their contexts into batch
        if len(self.additional_column_names) > 0:
            for additional_colname in self.additional_column_names:
                context[additional_colname] = [
                    row[additional_colname] for row in raw_batch
                ]
        return context

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        """
        Aggregates a batch of output data (predictions, scores, targets/true labels
        and loss).

        Args:
            n_batches (int): number of current batch
            preds (torch.Tensor): predictions of current batch
            targets (torch.Tensor): targets of current batch
            scores (torch.Tensor): scores of current batch
            loss (double): average loss of current batch
            m_input (Tuple[torch.Tensor, ...]): model inputs of current batch
            context (Dict[str, Any]): any additional context data, it could be
                either a list of data which maps to each example, or a single value
                for the batch
        """
        self.n_batches = n_batches
        self.aggregate_preds(preds, context)
        self.aggregate_targets(targets, context)

        # if we are running in memory efficient mode, then don't store all the scores
        # This list in general is the most memory hungry of all the data structures.
        # For a problem with 10K classes, we store 10K floats for every instance in
        # the epoch. This is bad.
        if not self.is_memory_efficient:
            self.aggregate_scores(scores)

        for key, val in context.items():
            if not (isinstance(val, torch.Tensor) or isinstance(val, List)):
                continue
            if key not in self.all_context:
                self.all_context[key] = []
            self.aggregate_data(self.all_context[key], val)
        # some loss functions (eg: in NewBertRegressionTask) return a tensor
        # convert tensor to float
        if loss is not None:
            self.all_loss.append(float(loss))
        # append the size of the first tensor (should be the batch size)
        if isinstance(m_input, Generator):
            # first element without updating the initial iterator
            first = tee(m_input, 1)
            self.batch_size.append(len(first))
        else:
            self.batch_size.append(len(m_input[0]))

    def calculate_metric(self):
        # If we are running in memory efficient mode, then scores in
        # LabelPrediction should be an empty list
        label_predictions = [
            LabelPrediction(scores, pred, expect)
            for scores, pred, expect in zip_longest(
                self.all_scores, self.all_preds, self.all_targets, fillvalue=[]
            )
        ]
        return compute_classification_metrics(
            label_predictions,
            self.label_names,
            self.calculate_loss(),
            # Compute soft-metrics only if self.is_memory_efficient is False
            average_precisions=(not self.is_memory_efficient),
            recall_at_precision_thresholds=self.recall_at_precision_thresholds,
        )

    def predictions_to_report(self):
        """
        Generate human readable predictions
        """
        return [self.label_names[pred] for pred in self.all_preds]

    def targets_to_report(self):
        """
        Generate human readable targets
        """
        return [self.label_names[target] for target in self.all_targets]

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


class MultiLabelClassificationMetricReporter(ClassificationMetricReporter):
    class Config(ClassificationMetricReporter.Config):
        pass

    def calculate_metric(self):
        return compute_multi_label_classification_metrics(
            [
                LabelListPrediction(scores, pred, expect)
                for scores, pred, expect in zip(
                    self.all_scores, self.all_preds, self.all_targets
                )
            ],
            self.label_names,
            self.calculate_loss(),
            recall_at_precision_thresholds=self.recall_at_precision_thresholds,
        )

    def predictions_to_report(self):
        """
        Generate human readable predictions
        """
        return [
            [
                self.label_names[pred_idx]
                for pred_idx, pred in enumerate(predictions)
                if pred == 1
            ]
            for predictions in self.all_preds
        ]

    def targets_to_report(self):
        """
        Generate human readable targets
        """
        return [
            [self.label_names[target] for target in targets if target != -1]
            for targets in self.all_targets
        ]
