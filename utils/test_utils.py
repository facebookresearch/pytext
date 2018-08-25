#!/usr/bin/env python3


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
