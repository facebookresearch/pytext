#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
import string
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from pytext.common.constants import Stage
from pytext.metric_reporters.channel import Channel, ConsoleChannel, FileChannel
from pytext.metric_reporters.metric_reporter import MetricReporter
from pytext.metrics.squad_metrics import SquadMetrics
from pytext.utils.data import merge_token_labels_to_slot, parse_slot_string
from scipy.optimize import linear_sum_assignment


class MultiSpanQAFileChannel(FileChannel):
    def get_title(self, context_keys=()):
        return (
            "index",
            "question",
            "document",
            "predicted_answers",
            "predicted_ans_starts",
            "predicted_ans_ends",
            "true_answers",
            "true_ans_starts",
            "true_ans_ends",
        )

    def gen_content(self, metrics, loss, preds, targets, scores, contexts, *args):
        pred_answers, pred_starts, pred_ends = preds
        true_answers, true_starts, true_ends = targets
        for i in range(len(pred_answers)):
            yield [
                contexts[MultiSpanQAMetricReporter.ROW_INDEX][i],
                contexts[MultiSpanQAMetricReporter.QUESTION_COLUMN][i],
                contexts[MultiSpanQAMetricReporter.DOC_COLUMN][i],
                pred_answers[i],
                pred_starts[i],
                pred_ends[i],
                true_answers[i],
                true_starts[i],
                true_ends[i],
            ]


class MultiSpanQAMetricReporter(MetricReporter):
    QUESTION_COLUMN = "question"
    ANSWERS_COLUMN = "labels"
    DOC_COLUMN = "document"
    ROW_INDEX = "row_index"

    class Config(MetricReporter.Config):
        pass

    @classmethod
    def from_config(cls, config, *args, tensorizers=None, **kwargs):
        return cls(
            channels=[
                ConsoleChannel(),
                MultiSpanQAFileChannel((Stage.TEST,), config.output_path),
            ],
            tensorizer=tensorizers["tokens"],
        )

    def __init__(
        self,
        channels: List[Channel],
        tensorizer=None,
    ) -> None:
        super().__init__(channels)
        self.channels = channels
        self.tensorizer = tensorizer

    def _reset(self):
        super()._reset()
        self.all_pred_answers: List = []
        self.all_pred_starts: List = []
        self.all_pred_ends: List = []
        self.all_target_answers: List = []
        self.all_target_starts: List = []
        self.all_target_ends: List = []

        self.all_preds = (
            self.all_pred_answers,
            self.all_pred_starts,
            self.all_pred_ends,
        )
        self.all_targets = (
            self.all_target_answers,
            self.all_target_starts,
            self.all_target_ends,
        )
        self.all_context: Dict = {}
        self.all_loss: List = []
        self.batch_size: List = []
        self.n_batches = 0

    def _add_answer_batch_stats(self, m_input, m_labels, **contexts):
        # For BERT, doc_tokens = concatenated tokens from question and document.
        doc_tokens = m_input[0][0]
        ans_list = []
        ans_starts_list = []
        ans_ends_list = []
        for m_labels, tokens, doc_str in zip(
            m_labels,
            doc_tokens,
            contexts[self.DOC_COLUMN],
        ):
            ans, ans_starts, ans_ends = self._unnumberize(
                m_labels, tokens.tolist(), doc_str
            )
            ans_list.append(ans)
            ans_starts_list.append(ans_starts)
            ans_ends_list.append(ans_ends)

        return ans_list, ans_starts_list, ans_ends_list

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **contexts
    ):  # contexts object is the dict returned by self.batch_context().
        super().add_batch_stats(
            n_batches, preds, targets, scores, loss, m_input, **contexts
        )

        # for preds
        ans_list, ans_starts_list, ans_ends_list = self._add_answer_batch_stats(
            m_input, preds, **contexts
        )
        self.aggregate_data(self.all_pred_answers, ans_list)
        self.aggregate_data(self.all_pred_starts, ans_starts_list)
        self.aggregate_data(self.all_pred_ends, ans_ends_list)

        # for targets
        ans_list, ans_starts_list, ans_ends_list = self._add_answer_batch_stats(
            m_input, targets, **contexts
        )
        self.aggregate_data(self.all_target_answers, ans_list)
        self.aggregate_data(self.all_target_starts, ans_starts_list)
        self.aggregate_data(self.all_target_ends, ans_ends_list)

    def aggregate_preds(self, new_batch, context=None):
        pass

    def aggregate_targets(self, new_batch, context=None):
        pass

    def batch_context(self, raw_batch, batch):
        context = super().batch_context(raw_batch, batch)
        context[self.ROW_INDEX] = [row[self.ROW_INDEX] for row in raw_batch]
        context[self.QUESTION_COLUMN] = [row[self.QUESTION_COLUMN] for row in raw_batch]
        context[self.ANSWERS_COLUMN] = [row[self.ANSWERS_COLUMN] for row in raw_batch]
        context[self.DOC_COLUMN] = [row[self.DOC_COLUMN] for row in raw_batch]
        return context

    def calculate_metric(self):
        self.all_preds = (
            self.all_pred_answers,
            self.all_pred_starts,
            self.all_pred_ends,
        )
        self.all_targets = (
            self.all_target_answers,
            self.all_target_starts,
            self.all_target_ends,
        )

        exact_matches, f1_scores_sum = self._compute_exact_matches_and_f1_scores_sum(
            self.all_pred_answers,
            self.all_target_answers,
        )
        count = len(self.all_pred_answers)
        metrics = SquadMetrics(
            exact_matches=100.0 * exact_matches / count,
            f1_score=100.0 * f1_scores_sum / count,
            num_examples=count,
            classification_metrics=None,
        )
        return metrics

    def get_model_select_metric(self, metric: SquadMetrics):
        return metric.f1_score

    def _unnumberize(self, preds, tokens, doc_str):
        """
        We re-tokenize and re-numberize the raw context (doc_str) here to get doc_tokens to get
        access to start_idx and end_idx mappings.  At this point, ans_token_start is the start index
        of the answer within tokens and ans_token_end is the end index. We calculate the offset of doc_tokens
        within tokens.
        Then we find the start_idx and end_idx
        as well as the corresponding span in the raw text using the answer token indices.
        """
        # start_idx and end_idx are lists of char start and end positions in doc_str.
        doc_tokens, start_idxs, end_idxs = self.tensorizer._lookup_tokens(doc_str)

        # find the offsets of doc_tokens in tokens
        try:
            offset_end = tokens.index(self.tensorizer.vocab.get_pad_index()) - 1
        except ValueError:
            offset_end = len(tokens) - 1
        offset_start = list(
            map(
                lambda x: tokens[x:offset_end] == doc_tokens[: offset_end - x],
                range(offset_end),
            )
        ).index(True)

        # find each answer's char idxs and strings as well
        pred_labels = self._process_pred(preds[offset_start:offset_end])
        token_range = list(zip(start_idxs, end_idxs))

        pred_slots = parse_slot_string(
            merge_token_labels_to_slot(
                token_range,
                pred_labels,
                self.tensorizer.use_bio_labels,
            )
        )
        ans_strs = []
        ans_start_char_idxs = []
        ans_end_char_idxs = []
        for slot in pred_slots:
            # if its not an answer span, skip
            if slot.label in map(
                str,
                [
                    self.tensorizer.labels_vocab.pad_token,
                    self.tensorizer.labels_vocab.unk_token,
                ],
            ):
                continue
            ans_strs.append(doc_str[slot.start : slot.end])
            ans_start_char_idxs.append(slot.start)
            ans_end_char_idxs.append(slot.end)

        return ans_strs, ans_start_char_idxs, ans_end_char_idxs

    def _process_pred(self, pred: List[int]) -> List[str]:
        """pred is a list of token label index"""
        return [self.tensorizer.labels_vocab[p] for p in pred]

    def _compute_exact_matches_and_f1_scores_sum(
        self,
        pred_answer_list: List[List[str]],
        target_answers_list: List[List[str]],
    ) -> Tuple[float, float]:
        exact_matches = 0.0
        f1_scores_sum = 0.0
        for pred_answers, target_answers in zip(
            pred_answer_list,
            target_answers_list,
        ):
            pred_spans, pred_bags = self._answer_to_bags(pred_answers)
            target_spans, target_bags = self._answer_to_bags(target_answers)
            if set(pred_spans) == set(target_spans) and len(pred_spans) == len(
                target_spans
            ):
                exact_matches += 1.0

            f1_per_bag = self._align_bags(pred_bags, target_bags)
            if len(f1_per_bag):
                # avoid NaNs
                f1 = np.mean(f1_per_bag)
                f1_scores_sum += round(f1, 2)

        return exact_matches, f1_scores_sum

    # The following functions are copied from DROP's evaluation script.
    # https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/tools/drop.py

    def _answer_to_bags(
        self, answer: Union[str, List[str], Tuple[str, ...]]
    ) -> Tuple[List[str], List[Set[str]]]:
        if isinstance(answer, (list, tuple)):
            raw_spans = answer
        else:
            raw_spans = [answer]
        normalized_spans: List[str] = []
        token_bags = []
        for raw_span in raw_spans:
            normalized_span = self._normalize_answer(raw_span)
            normalized_spans.append(normalized_span)
            token_bags.append(set(normalized_span.split()))
        return normalized_spans, token_bags

    def _normalize_answer(self, s: str):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _align_bags(
        self, predicted: List[Set[str]], gold: List[Set[str]]
    ) -> List[float]:
        """
        Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
        between them and gets maximum metric values over all the answers.
        """
        scores = np.zeros([len(gold), len(predicted)])
        for gold_index, gold_item in enumerate(gold):
            for pred_index, pred_item in enumerate(predicted):
                scores[gold_index, pred_index] = self._compute_f1(pred_item, gold_item)
        row_ind, col_ind = linear_sum_assignment(-scores)

        max_scores = np.zeros([max(len(gold), len(predicted))])
        for row, column in zip(row_ind, col_ind):
            max_scores[row] = max(max_scores[row], scores[row, column])
        return max_scores

    def _compute_f1(self, predicted_bag: Set[str], gold_bag: Set[str]) -> float:
        intersection = len(gold_bag.intersection(predicted_bag))
        if not predicted_bag:
            precision = 1.0
        else:
            precision = intersection / float(len(predicted_bag))
        if not gold_bag:
            recall = 1.0
        else:
            recall = intersection / float(len(gold_bag))
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if not (precision == 0.0 and recall == 0.0)
            else 0.0
        )
        return f1
