#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import re
from collections import Counter
from typing import Dict, List, NamedTuple

from pytext.common.constants import DatasetFieldName, SpecialTokens, Stage
from pytext.data import CommonMetadata
from pytext.metrics import (
    AllConfusions,
    Confusions,
    LabelPrediction,
    PRF1Metrics,
    compute_classification_metrics,
    compute_multi_label_multi_class_soft_metrics,
)
from pytext.metrics.intent_slot_metrics import (
    Node,
    NodesPredictionPair,
    Span,
    compute_prf1_metrics,
)
from pytext.utils.data import merge_token_labels_to_slot, parse_slot_string

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


NAN_LABELS = [SpecialTokens.UNK, SpecialTokens.PAD]


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


class MultiLabelSequenceTaggingMetricReporter(MetricReporter):
    def __init__(self, label_names, pad_idx, channels, label_vocabs=None):
        self.label_names = label_names
        self.pad_idx = pad_idx
        self.label_vocabs = label_vocabs
        super().__init__(channels)

    @classmethod
    def from_config(cls, config, tensorizers):
        return MultiLabelSequenceTaggingMetricReporter(
            channels=[ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
            label_names=tensorizers.keys(),
            pad_idx=[v.pad_idx for _, v in tensorizers.items()],
            label_vocabs=[v.vocab._vocab for _, v in tensorizers.items()],
        )

    def aggregate_tuple_data(self, all_data, new_batch):
        assert isinstance(new_batch, tuple)
        # num_label_set * bsz * ...
        data = [self._make_simple_list(d) for d in new_batch]
        # convert to bsz * num_label_set * ...
        for d in zip(*data):
            all_data.append(d)

    def aggregate_preds(self, batch_preds, batch_context=None):
        self.aggregate_tuple_data(self.all_preds, batch_preds)

    def aggregate_targets(self, batch_targets, batch_context=None):
        self.aggregate_tuple_data(self.all_targets, batch_targets)

    def aggregate_scores(self, batch_scores):
        self.aggregate_tuple_data(self.all_scores, batch_scores)

    def calculate_metric(self):
        list_score_pred_expect = []
        for label_idx, _ in enumerate(self.label_names):
            list_score_pred_expect.append(
                list(
                    itertools.chain.from_iterable(
                        (
                            LabelPrediction(s, p, e)
                            for s, p, e in zip(
                                scores[label_idx], pred[label_idx], expect[label_idx]
                            )
                            if e != self.pad_idx[label_idx]
                        )
                        for scores, pred, expect in zip(
                            self.all_scores, self.all_preds, self.all_targets
                        )
                    )
                )
            )

        metrics = compute_multi_label_multi_class_soft_metrics(
            list_score_pred_expect, self.label_names, self.label_vocabs
        )
        return metrics

    def batch_context(self, raw_batch, batch):
        return {}

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.average_overall_precision


class SequenceTaggingMetricReporter(MetricReporter):
    def __init__(self, label_names, pad_idx, channels):
        super().__init__(channels)
        self.label_names = label_names
        self.pad_idx = pad_idx

    @classmethod
    def from_config(cls, config, tensorizer):
        return SequenceTaggingMetricReporter(
            channels=[ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
            label_names=list(tensorizer.vocab),
            pad_idx=tensorizer.pad_idx,
        )

    def calculate_metric(self):
        return compute_classification_metrics(
            list(
                itertools.chain.from_iterable(
                    (
                        LabelPrediction(s, p, e)
                        for s, p, e in zip(scores, pred, expect)
                        if e != self.pad_idx
                    )
                    for scores, pred, expect in zip(
                        self.all_scores, self.all_preds, self.all_targets
                    )
                )
            ),
            self.label_names,
            self.calculate_loss(),
        )

    def batch_context(self, raw_batch, batch):
        return {}

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.accuracy


class Span(NamedTuple):
    label: str
    start: int
    end: int


def convert_bio_to_spans(bio_sequence: List[str]) -> List[Span]:
    """
    Process the output and convert to spans for evaluation.
    """
    spans = []  # (label, startindex, endindex)
    cur_start = None
    cur_label = None
    N = len(bio_sequence)
    for t in range(N + 1):
        if (cur_start is not None) and (t == N or re.search("^[BO]", bio_sequence[t])):
            assert cur_label is not None
            spans.append(Span(cur_label, cur_start, t))
            cur_start = None
            cur_label = None
        if t == N:
            continue
        assert bio_sequence[t]
        if bio_sequence[t][0] not in ("B", "I", "O"):
            bio_sequence[t] = "O"
        if bio_sequence[t].startswith("B"):
            cur_start = t
            cur_label = re.sub("^B-?", "", bio_sequence[t]).strip()
        if bio_sequence[t].startswith("I"):
            if cur_start is None:
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                return convert_bio_to_spans(newseq)
            continuation_label = re.sub("^I-?", "", bio_sequence[t])
            if continuation_label != cur_label:
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                return convert_bio_to_spans(newseq)

    # should have exited for last span ending at end by now
    assert cur_start is None
    return spans


class NERMetricReporter(MetricReporter):
    def __init__(
        self,
        label_names: List[str],
        pad_idx: int,
        channels: List[Channel],
        use_bio_labels: bool = True,
    ) -> None:
        super().__init__(channels)
        self.label_names = label_names
        self.use_bio_labels = use_bio_labels
        self.pad_idx = pad_idx
        assert self.use_bio_labels

    @classmethod
    def from_config(cls, config, tensorizer):
        return WordTaggingMetricReporter(
            channels=[ConsoleChannel()],
            label_names=list(tensorizer.vocab),
            pad_idx=tensorizer.pad_idx,
        )

    def calculate_metric(self) -> PRF1Metrics:
        all_confusions = AllConfusions()
        for pred, expect in zip(self.all_preds, self.all_targets):
            pred_seq, expect_seq = [], []
            for p, e in zip(pred, expect):
                if e != self.pad_idx:
                    pred_seq.append(self.label_names[p])
                    expect_seq.append(self.label_names[e])
            expect_spans = convert_bio_to_spans(expect_seq)
            pred_spans = convert_bio_to_spans(pred_seq)

            expect_spans_set = set(expect_spans)
            pred_spans_set = set(pred_spans)

            true_positive = expect_spans_set & pred_spans_set
            false_positive = pred_spans_set - expect_spans_set
            false_negative = expect_spans_set - pred_spans_set
            all_confusions.confusions += Confusions(
                TP=len(true_positive), FP=len(false_positive), FN=len(false_negative)
            )
            for span in true_positive:
                all_confusions.per_label_confusions.update(span.label, "TP", 1)
            for span in false_positive:
                all_confusions.per_label_confusions.update(span.label, "FP", 1)
            for span in false_negative:
                all_confusions.per_label_confusions.update(span.label, "FN", 1)

        return all_confusions.compute_metrics()

    def batch_context(self, raw_batch, batch):
        return {}

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.micro_scores.f1
