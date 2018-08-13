#!/usr/bin/env python3
import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from pytext.utils import data_utils
from typing import Tuple


class ResultRow:
    def __init__(self, name, metrics_dict):
        self.name = name
        for m_name, m_val in metrics_dict.items():
            setattr(self, m_name, m_val)


class ResultTable:
    def __init__(self, metrics, class_names, labels, preds):
        self.rows = []
        for i, class_n in enumerate(class_names):
            metrics_dict = {}
            metrics_dict["num_samples"] = int(metrics[3][i])
            metrics_dict["num_correct"] = sum(
                int(label) == i and int(label) == int(preds[j])
                for j, label in enumerate(labels)
            )
            metrics_dict["precision"] = metrics[0][i]
            metrics_dict["recall"] = metrics[1][i]
            metrics_dict["f1"] = metrics[2][i]
            self.rows.append(ResultRow(class_n, metrics_dict))


def get_all_metrics(preds, labels_idx, class_names) -> Tuple[ResultTable, ResultRow]:
    metrics = precision_recall_fscore_support(
        labels_idx, preds, labels=range(0, len(class_names))
    )
    weighted_metrics = precision_recall_fscore_support(
        labels_idx, preds, labels=range(0, len(class_names)), average="weighted"
    )
    accuracy = accuracy_score(preds, labels_idx)
    result_table = ResultTable(metrics, class_names, labels_idx, preds)
    aggregate_metrics = ResultRow(
        "total",
        {
            "weighted_precision": weighted_metrics[0],
            "weighted_recall": weighted_metrics[1],
            "weighted_f1": weighted_metrics[2],
            "accuracy": accuracy,
        },
    )
    return result_table, aggregate_metrics


def summarize(tokens_length, tokenized_text, labels):
    # ToDo: Utilize the BIO information when going from token labels to span
    # labels instead of the greedy approach performed below
    tokens = []
    token_ranges = []
    for t, t_range in tokenized_text:
        tokens.append(t)
        token_ranges.append(t_range)
    assert len(tokens) == tokens_length
    assert len(token_ranges) == tokens_length
    assert len(labels) == tokens_length
    summary_list = []
    begin = token_ranges[0][0]
    end = token_ranges[0][1]

    for i in range(1, tokens_length):
        # Extend
        if labels[i] == labels[i - 1] and labels[i] != "NoLabel":
            end = token_ranges[i][1]

        # Update and start new
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] != "NoLabel")
            and (labels[i - 1] != "NoLabel")
        ):
            summary_list.append(":".join([str(begin), str(end), labels[i - 1]]))
            begin = token_ranges[i][0]
            end = token_ranges[i][1]

        # Update and skip
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] == "NoLabel")
            and (labels[i - 1] != "NoLabel")
        ):
            summary_list.append(":".join([str(begin), str(end), labels[i - 1]]))

        # Skip
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] != "NoLabel")
            and (labels[i - 1] == "NoLabel")
        ):
            begin = token_ranges[i][0]
            end = token_ranges[i][1]

    # Take last token into account
    if labels[-1] != "NoLabel":
        summary_list.append(":".join([str(begin), str(end), labels[-1]]))
    return ",".join(summary_list)


def count_chunk_match(predictions, labels):
    """
    for each prediction and label pair, count matched chunks
    return a json format string
    """
    chunk_match_dict = defaultdict(float)
    if predictions == "" or labels == "":
        return json.dumps(chunk_match_dict)

    label_list = data_utils.parse_slot_string(labels)
    predictions_list = data_utils.parse_slot_string(predictions)

    for prediction in predictions_list:
        for gold_label in label_list:
            if (
                (abs(gold_label.start - prediction.start) <= 2)
                and (abs(gold_label.end - prediction.end) <= 2)
                and prediction.label == gold_label.label
            ):
                chunk_match_dict[prediction.label] += 1
                break
    return json.dumps(chunk_match_dict)
