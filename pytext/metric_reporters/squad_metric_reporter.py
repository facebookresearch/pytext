#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
import string
from collections import Counter
from typing import Dict, List

import numpy as np
from pytext.common.constants import Stage
from pytext.metric_reporters.channel import Channel, ConsoleChannel, FileChannel
from pytext.metric_reporters.metric_reporter import MetricReporter
from pytext.metrics.squad_metrics import SquadMetrics


class SquadFileChannel(FileChannel):
    def get_title(self, context_keys=()):
        return (
            "index",
            "ques",
            "doc",
            "predicted_answer",
            "true_answers",
            "predicted_start_pos",
            "predicted_end_pos",
            "true_start_pos",
            "true_end_pos",
            "start_pos_scores",
            "end_pos_scores",
            "predicted_has_answer",
            "true_has_answer",
            "has_answer_scores",
        )

    def gen_content(self, metrics, loss, preds, targets, scores, contexts, *args):
        pred_answers, pred_start_pos, pred_end_pos, pred_has_answer = preds
        true_answers, true_start_pos, true_end_pos, true_has_answer = targets
        start_pos_scores, end_pos_scores, has_answer_scores = scores
        for i in range(len(pred_answers)):
            yield [
                contexts[SquadMetricReporter.ROW_INDEX][i],
                contexts[SquadMetricReporter.QUES_COLUMN][i],
                contexts[SquadMetricReporter.DOC_COLUMN][i],
                pred_answers[i],
                true_answers[i],
                pred_start_pos[i],
                pred_end_pos[i],
                true_start_pos[i],
                true_end_pos[i],
                start_pos_scores[i],
                end_pos_scores[i],
                pred_has_answer[i],
                true_has_answer[i],
                has_answer_scores[i],
            ]


class SquadMetricReporter(MetricReporter):
    QUES_COLUMN = "question"
    ANSWERS_COLUMN = "answers"
    DOC_COLUMN = "doc"
    ROW_INDEX = "id"

    class Config(MetricReporter.Config):
        n_best_size: int = 5
        max_answer_length: int = 16
        ignore_impossible: bool = True
        false_label: str = "False"

    @classmethod
    def from_config(cls, config, *args, tensorizers=None, **kwargs):
        return cls(
            channels=[
                ConsoleChannel(),
                SquadFileChannel((Stage.TEST,), config.output_path),
            ],
            n_best_size=config.n_best_size,
            max_answer_length=config.max_answer_length,
            ignore_impossible=config.ignore_impossible,
            has_answer_labels=tensorizers["has_answer"].vocab._vocab,
            tensorizer=tensorizers["squad_input"],
            false_label=config.false_label,
        )

    def __init__(
        self,
        channels: List[Channel],
        n_best_size: int,
        max_answer_length: int,
        ignore_impossible: bool,
        has_answer_labels: List[str],
        tensorizer=None,
        false_label=Config.false_label,
    ) -> None:
        super().__init__(channels)
        self.channels = channels
        self.tensorizer = tensorizer
        self.ignore_impossible = ignore_impossible
        self.has_answer_labels = has_answer_labels
        self.false_label = false_label
        self.false_idx = 1 if has_answer_labels[1] == false_label else 0
        self.true_idx = 1 - self.false_idx

    def _reset(self):
        self.all_start_pos_preds: List = []
        self.all_start_pos_targets: List = []
        self.all_start_pos_scores: List = []
        self.all_end_pos_preds: List = []
        self.all_end_pos_targets: List = []
        self.all_end_pos_scores: List = []
        self.all_has_answer_targets: List = []
        self.all_has_answer_preds: List = []
        self.all_has_answer_scores: List = []

        self.all_preds = (
            self.all_start_pos_preds,
            self.all_end_pos_preds,
            self.all_has_answer_preds,
        )
        self.all_targets = (
            self.all_start_pos_targets,
            self.all_end_pos_targets,
            self.all_has_answer_targets,
        )
        self.all_scores = (
            self.all_start_pos_scores,
            self.all_end_pos_scores,
            self.all_has_answer_scores,
        )
        self.all_context: Dict = {}
        self.all_loss: List = []
        self.all_pred_answers: List = []
        # self.all_true_answers: List = []
        self.batch_size: List = []
        self.n_batches = 0

    def _add_decoded_answer_batch_stats(self, m_input, preds, **contexts):
        # For BERT, doc_tokens = concatenated tokens from question and document.
        doc_tokens = m_input[0]
        starts, ends, has_answers = preds
        pred_answers = [
            self._unnumberize(tokens[start : end + 1].tolist(), doc_str)
            for tokens, start, end, doc_str in zip(
                doc_tokens, starts, ends, contexts[self.DOC_COLUMN]
            )
        ]
        self.aggregate_data(self.all_pred_answers, pred_answers)

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **contexts
    ):  # contexts object is the dict returned by self.batch_context().
        super().add_batch_stats(
            n_batches, preds, targets, scores, loss, m_input, **contexts
        )
        self._add_decoded_answer_batch_stats(m_input, preds, **contexts)

    def aggregate_preds(self, new_batch, context=None):
        self.aggregate_data(self.all_start_pos_preds, new_batch[0])
        self.aggregate_data(self.all_end_pos_preds, new_batch[1])
        self.aggregate_data(self.all_has_answer_preds, new_batch[2])

    def aggregate_targets(self, new_batch, context=None):
        self.aggregate_data(self.all_start_pos_targets, new_batch[0])
        self.aggregate_data(self.all_end_pos_targets, new_batch[1])
        self.aggregate_data(self.all_has_answer_targets, new_batch[2])

    def aggregate_scores(self, new_batch):
        self.aggregate_data(self.all_start_pos_scores, new_batch[0])
        self.aggregate_data(self.all_end_pos_scores, new_batch[1])
        self.aggregate_data(self.all_has_answer_scores, new_batch[2])

    def batch_context(self, raw_batch, batch):
        context = super().batch_context(raw_batch, batch)
        context[self.ROW_INDEX] = [row[self.ROW_INDEX] for row in raw_batch]
        context[self.QUES_COLUMN] = [row[self.QUES_COLUMN] for row in raw_batch]
        context[self.ANSWERS_COLUMN] = [row[self.ANSWERS_COLUMN] for row in raw_batch]
        context[self.DOC_COLUMN] = [row[self.DOC_COLUMN] for row in raw_batch]
        return context

    def calculate_metric(self):
        all_rows = zip(
            self.all_context[self.ROW_INDEX],
            self.all_context[self.ANSWERS_COLUMN],
            self.all_context[self.QUES_COLUMN],
            self.all_context[self.DOC_COLUMN],
            self.all_pred_answers,
            self.all_start_pos_preds,
            self.all_end_pos_preds,
            self.all_has_answer_preds,
            self.all_start_pos_targets,
            self.all_end_pos_targets,
            self.all_has_answer_targets,
            self.all_start_pos_scores,
            self.all_end_pos_scores,
            self.all_has_answer_scores,
        )

        all_rows_dict = {}
        for row in all_rows:
            try:
                all_rows_dict[row[0]].append(row)
            except KeyError:
                all_rows_dict[row[0]] = [row]

        all_rows = []
        for rows in all_rows_dict.values():
            argmax = np.argmax([row[11] + row[12] for row in rows])
            all_rows.append(rows[argmax])

        sorted(all_rows, key=lambda x: int(x[0]))

        (
            self.all_context[self.ROW_INDEX],
            self.all_context[self.ANSWERS_COLUMN],
            self.all_context[self.QUES_COLUMN],
            self.all_context[self.DOC_COLUMN],
            self.all_pred_answers,
            self.all_start_pos_preds,
            self.all_end_pos_preds,
            self.all_has_answer_preds,
            self.all_start_pos_targets,
            self.all_end_pos_targets,
            self.all_has_answer_targets,
            self.all_start_pos_scores,
            self.all_end_pos_scores,
            self.all_has_answer_scores,
        ) = zip(*all_rows)

        exact_matches, count = self._compute_exact_matches(
            self.all_pred_answers,
            self.all_context[self.ANSWERS_COLUMN],
            self.all_has_answer_preds,
            self.all_has_answer_targets,
        )
        f1_score = self._compute_f1_score(
            self.all_pred_answers,
            self.all_context[self.ANSWERS_COLUMN],
            self.all_has_answer_preds,
            self.all_has_answer_targets,
        )
        self.all_preds = (
            self.all_pred_answers,
            self.all_start_pos_preds,
            self.all_end_pos_preds,
            self.all_has_answer_preds,
        )
        self.all_targets = (
            self.all_context[self.ANSWERS_COLUMN],
            self.all_start_pos_targets,
            self.all_end_pos_targets,
            self.all_has_answer_targets,
        )
        self.all_scores = (
            self.all_start_pos_scores,
            self.all_end_pos_scores,
            self.all_has_answer_scores,
        )
        metrics = SquadMetrics(
            exact_matches=100.0 * exact_matches / count,
            f1_score=f1_score,
            num_examples=count,
        )
        return metrics

    def get_model_select_metric(self, metric: SquadMetrics):
        return metric.f1_score

    def _compute_exact_matches(
        self,
        pred_answer_list,
        target_answers_list,
        pred_has_answer_list,
        target_has_answer_list,
    ):
        exact_matches = 0
        for pred_answer, target_answers, pred_has_answer, target_has_answer in zip(
            pred_answer_list,
            target_answers_list,
            pred_has_answer_list,
            target_has_answer_list,
        ):
            if not self.ignore_impossible:
                if pred_has_answer != target_has_answer:
                    continue
                if pred_has_answer == self.false_idx:
                    exact_matches += 1
                    continue
            pred = self._normalize_answer(pred_answer)
            for answer in target_answers:
                true = self._normalize_answer(answer)
                if pred == true:
                    exact_matches += 1
                    break
        return exact_matches, len(pred_answer_list)

    def _compute_f1_score(
        self,
        pred_answer_list,
        target_answers_list,
        pred_has_answer_list,
        target_has_answer_list,
    ):
        f1_scores_sum = 0.0
        for pred_answer, target_answers, pred_has_answer, target_has_answer in zip(
            pred_answer_list,
            target_answers_list,
            pred_has_answer_list,
            target_has_answer_list,
        ):
            if not self.ignore_impossible:
                if pred_has_answer != target_has_answer:
                    continue
                if pred_has_answer == self.false_idx:
                    f1_scores_sum += 1.0
                    continue
            f1_scores_sum += max(
                self._compute_f1_per_answer(answer, pred_answer)
                for answer in target_answers
            )
        return 100.0 * f1_scores_sum / len(pred_answer_list)

    def _unnumberize(self, ans_tokens, doc_str):
        """
        Tokens is the span of token ids that the model predicted. We re-tokenize
        and re-numberize the raw context (doc_str) here to get doc_tokens to get
        access to start_idx and end_idx mappings.  At this point, ans_tokens is
        a sub-list of doc_tokens (hopefully, if the model predicted a span in
        the context). Then we find tokens inside doc_tokens, and return the
        corresponding span in the raw text using the idx mapping.
        """
        # start_idx and end_idx are lists of char start and end positions in doc_str.
        doc_tokens, start_idx, end_idx = self.tensorizer._lookup_tokens(doc_str)
        doc_tokens = list(doc_tokens)
        num_ans_tokens = len(ans_tokens)
        answer_str = ""
        for doc_token_idx in range(len(doc_tokens) - num_ans_tokens):
            if doc_tokens[doc_token_idx : doc_token_idx + num_ans_tokens] == ans_tokens:
                start_char_idx = start_idx[doc_token_idx]
                end_char_idx = end_idx[doc_token_idx + num_ans_tokens - 1]
                answer_str = doc_str[start_char_idx:end_char_idx]
                break
        return answer_str

    # The following three functions are copied from Squad's evaluation script.
    # https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

    def _normalize_answer(self, s):
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

    def _get_tokens(self, s):
        if not s:
            return []
        return self._normalize_answer(s).split()

    def _compute_f1_per_answer(self, a_gold, a_pred):
        gold_toks = self._get_tokens(a_gold)
        pred_toks = self._get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
