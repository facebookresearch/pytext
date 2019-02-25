#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import Counter
from typing import List

import torch
from pytext.common.constants import DatasetFieldName, Stage
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.metrics import AllConfusions, Confusions
from pytext.metrics.intent_slot_metrics import (
    Node,
    NodesPredictionPair,
    Span,
    compute_prf1_metrics,
)
from pytext.utils.data_utils import parse_slot_string
from pytext.utils.test_utils import merge_token_labels_to_slot

from .channel import Channel, ConsoleChannel, DetailFileChannel, FileChannel
from .metric_reporter import MetricReporter


def get_slots(word_names):
    slots = {
        Node(label=slot.label, span=Span(slot.start, slot.end))
        for slot in parse_slot_string(word_names)
    }
    return Counter(slots)


class WordTaggingMetricReporter(MetricReporter):
    def __init__(
        self, label_names: List[str], use_bio_labels: bool, channels: List[Channel]
    ) -> None:
        super().__init__(channels)
        self.label_names = label_names
        self.use_bio_labels = use_bio_labels

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls(
            meta.target.vocab.itos,
            meta.target.use_bio_labels,
            [ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
        )

    def calculate_loss(self):
        total_loss = n_words = pos = 0
        for loss, batch_size in zip(self.all_loss, self.batch_size):
            num_words_in_batch = sum(
                self.all_context["seq_lens"][pos : pos + batch_size]
            )
            pos = pos + batch_size
            total_loss += loss * num_words_in_batch
            n_words += num_words_in_batch
        return total_loss / float(n_words)

    def process_pred(self, pred: List[int]) -> List[str]:
        """pred is a list of token label index
        """
        return [self.label_names[p] for p in pred]

    def calculate_metric(self):
        return compute_prf1_metrics(
            [
                NodesPredictionPair(
                    get_slots(
                        merge_token_labels_to_slot(
                            token_range,
                            self.process_pred(pred[0:seq_len]),
                            self.use_bio_labels,
                        )
                    ),
                    get_slots(slots_label),
                )
                for pred, seq_len, token_range, slots_label in zip(
                    self.all_preds,
                    self.all_context[DatasetFieldName.SEQ_LENS],
                    self.all_context[DatasetFieldName.TOKEN_RANGE],
                    self.all_context[DatasetFieldName.RAW_WORD_LABEL],
                )
            ]
        )[1]

    def get_model_select_metric(self, metrics):
        return metrics.micro_scores.f1


class BertWordTaggingMetricReporter(MetricReporter):
    class Config(MetricReporter.Config):
        output_path: str = "/tmp/test_out.txt"
        use_crf: bool = False

    def __init__(
        self,
        use_crf: bool,
        label_names: List[str],
        use_bio_labels: bool,
        channels: List[Channel],
    ) -> None:
        super().__init__(channels)
        self.label_names = label_names
        self.use_bio_labels = use_bio_labels
        self.use_crf = use_crf

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls(
            config.use_crf,
            meta.target.vocab.itos,
            meta.target.use_bio_labels,
            [ConsoleChannel(), DetailFileChannel((Stage.TEST,), config.output_path)],
        )

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
        targets = self._strip_target(targets, scores.size()[-1])
        preds = self._strip_preds_or_scores(preds, context)
        scores = self._strip_preds_or_scores(scores, context)
        super().add_batch_stats(
            n_batches, preds, targets, scores, loss, m_input, **context
        )

    def _strip_preds_or_scores(self, new_batch, context):
        batch_list = []
        pad_mask = context["pad_mask"]
        token_range = context["token_range"]
        for i, row_tensor in enumerate(new_batch.split(1, dim=0)):
            pred = row_tensor.squeeze()
            if self.use_crf:
                num_token = len(token_range[i])
                pred = pred[:num_token]
            else:
                pad_mask_row = pad_mask[i, :]
                pred = pred[pad_mask_row]
                token_start = [r[0] for r in token_range[i]]
                token_start_idx = torch.tensor(token_start, device=pred.device).long()
                pred = pred.index_select(0, token_start_idx)
            batch_list.append(pred)
        return batch_list

    def _strip_target(self, new_batch, context):
        batch_list = []
        for row_tensor in new_batch.split(1, dim=0):
            row_tensor = row_tensor.squeeze()
            row_tensor = row_tensor[row_tensor < context]
            batch_list.append(row_tensor)
        return batch_list

    def calculate_metric(self):
        all_confusions = AllConfusions()
        true_positives = dict(
            zip(self.label_names[:-1], [0] * len(self.label_names[:-1]))
        )
        false_positives = dict(
            zip(self.label_names[:-1], [0] * len(self.label_names[:-1]))
        )
        false_negatives = dict(
            zip(self.label_names[:-1], [0] * len(self.label_names[:-1]))
        )

        for predicted_idx_example, expected_idx_example in zip(
            self.all_preds, self.all_targets
        ):
            for predicted_idx, expected_idx in zip(
                predicted_idx_example, expected_idx_example
            ):
                if predicted_idx == expected_idx:
                    true_positives[self.label_names[expected_idx]] += 1
                else:
                    false_negatives[self.label_names[expected_idx]] += 1
                    false_positives[self.label_names[predicted_idx]] += 1
        for label, count in true_positives.items():
            all_confusions.per_label_confusions.update(label, "TP", count)
        for label, count in false_positives.items():
            all_confusions.per_label_confusions.update(label, "FP", count)
        for label, count in false_negatives.items():
            all_confusions.per_label_confusions.update(label, "FN", count)

        all_confusions.confusions = Confusions(
            TP=sum(true_positives.values()),
            FP=sum(false_positives.values()),
            FN=sum(false_negatives.values()),
        )

        return all_confusions.compute_metrics()
