#!/usr/bin/env python3
from pytext.utils.data_utils import Slot


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


def strip_bio_prefix(label):
    if label.startswith(Slot.B_LABEL_PREFIX) or label.startswith(Slot.I_LABEL_PREFIX):
        label = label[len(Slot.B_LABEL_PREFIX) :]
    return label


def merge_token_labels_by_bio(token_ranges, labels):
    summary_list = []
    previous_B = None
    for i, label in enumerate(labels):
        # Take action only if the prefix is not i
        if not label.startswith(Slot.I_LABEL_PREFIX):
            # Label the previous chunk
            if previous_B is not None:
                begin = token_ranges[previous_B][0]
                end = token_ranges[i - 1][1]
                summary_list.append(
                    ":".join([str(begin), str(end), strip_bio_prefix(labels[i - 1])])
                )
            # Assign the begin location of new chunk
            if label.startswith(Slot.B_LABEL_PREFIX):
                previous_B = i
            else:  # label == Slot.NO_LABEL_SLOT
                previous_B = None

    # Take last token into account
    if previous_B is not None:
        begin = token_ranges[previous_B][0]
        end = token_ranges[-1][1]
        summary_list.append(
            ":".join([str(begin), str(end), strip_bio_prefix(labels[-1])])
        )

    return summary_list


def merge_token_labels_by_label(token_ranges, labels):
    # no bio prefix in labels
    begin = token_ranges[0][0]
    end = token_ranges[0][1]

    summary_list = []
    for i in range(1, len(labels)):
        # Extend
        if labels[i] == labels[i - 1] and labels[i] != Slot.NO_LABEL_SLOT:
            end = token_ranges[i][1]

        # Update and start new
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] != Slot.NO_LABEL_SLOT)
            and (labels[i - 1] != Slot.NO_LABEL_SLOT)
        ):
            summary_list.append(":".join([str(begin), str(end), labels[i - 1]]))
            begin = token_ranges[i][0]
            end = token_ranges[i][1]

        # Update and skip
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] == Slot.NO_LABEL_SLOT)
            and (labels[i - 1] != Slot.NO_LABEL_SLOT)
        ):
            summary_list.append(":".join([str(begin), str(end), labels[i - 1]]))

        # Skip
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] != Slot.NO_LABEL_SLOT)
            and (labels[i - 1] == Slot.NO_LABEL_SLOT)
        ):
            begin = token_ranges[i][0]
            end = token_ranges[i][1]

    # Take last token into account
    if labels[-1] != Slot.NO_LABEL_SLOT:
        summary_list.append(":".join([str(begin), str(end), labels[-1]]))

    return summary_list


def merge_token_labels_to_slot(tokenized_text, labels, use_bio_label=True):
    tokens = []
    token_ranges = []
    for t, t_range in tokenized_text:
        tokens.append(t)
        token_ranges.append(t_range)
    assert len(tokens) == len(labels)
    summary_list = (
        merge_token_labels_by_bio(token_ranges, labels)
        if use_bio_label
        else merge_token_labels_by_label(token_ranges, labels)
    )

    return ",".join(summary_list)
