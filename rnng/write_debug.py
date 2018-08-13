#!/usr/bin/env python3
import json
import sys
from collections import defaultdict, Counter
from collections import namedtuple

from pytext.rnng.annotation import Annotation, Slot, Token

DUMMY_SLOT_LABEL = "NoLabel"

Annotations = namedtuple(
    "Annotations",
    ["utterance", "predictions", "gold", "tokens", "intent_gold", "intent_pred"],
)

Gold_Anno = namedtuple("Gold_Anno", ["utterance", "gold", "tokens", "intent_gold"])


def prep_result_predictor(gold_file, pred_file, output_file):
    print("evaluating " + str(pred_file) + " against the gold file " + str(gold_file))
    num_instances = 0
    line_num = 0
    slot_labels = set()
    intent_labels = set()
    intent_labels_freq = Counter()

    annotations_list = []
    gold_anno_list = []
    num_invalid_tree = 0

    with open(pred_file) as pred_f, open(gold_file) as gold_f:
        line_num += 1
        for pred_line, gold_line in zip(pred_f, gold_f):
            num_instances += 1
            try:
                gold_tree = Annotation(gold_line).tree
            except ValueError:
                continue
            slot_gold_anno, slot_gold_labels, gold_intent = tree_to_predictions(
                gold_tree
            )
            intent_labels.add(gold_intent)
            slot_labels.update(slot_gold_labels)
            intent_labels_freq[gold_intent] += 1
            tokens = gold_tree.list_tokens()

            try:
                pred_tree = Annotation(pred_line).tree
                slot_pred_anno, slot_pred_labels, pred_intent = tree_to_predictions(
                    pred_tree
                )
                if pred_tree.list_tokens() != gold_tree.list_tokens():
                    raise ValueError(
                        "terminals for lines at {} are not the same: {} and {}".format(
                            line_num,
                            str(gold_tree.list_tokens()),
                            str(pred_tree.list_tokens()),
                        )
                    )

                slot_labels.update(slot_pred_labels)
                intent_labels.add(pred_intent)

                annotations_list.append(
                    Annotations(
                        utterance=" ".join(tokens),
                        predictions=slot_pred_anno,
                        gold=slot_gold_anno,
                        tokens=tokens,
                        intent_gold=gold_intent,
                        intent_pred=pred_intent,
                    )
                )
            except ValueError:
                num_invalid_tree += 1
                # predicted tree is not valid. save the gold annotation for now
                gold_anno_list.append(
                    Gold_Anno(
                        utterance=" ".join(tokens),
                        gold=slot_gold_anno,
                        tokens=tokens,
                        intent_gold=gold_intent,
                    )
                )
                continue

    print("Number of invalid trees: {}".format(num_invalid_tree))

    most_freq_intent = intent_labels_freq.most_common(1)[0][0]  # noqa: E501
    # add all invalid trees with the most frequent intent label and no slots
    for gn in gold_anno_list:
        annotations_list.append(
            Annotations(
                utterance=gn.utterance,
                predictions="",
                gold=gn.gold,
                tokens=gn.tokens,
                intent_gold=gn.intent_gold,
                intent_pred=most_freq_intent,
            )
        )

    slot_label_info = dict(enumerate(list(slot_labels)))

    intent_labels_list = list(intent_labels)

    with open(output_file, mode="w") as w_f:

        intent_names = "#{}".format(json.dumps(intent_labels_list))
        w_f.write(intent_names + "\n")

        if len(slot_labels) > 0:
            class_maxidx = max(  # noqa: C407
                [k for k in slot_label_info if isinstance(k, int)]
            )
            lab_arr = [slot_label_info[_i] for _i in range(class_maxidx + 1)]
        else:
            # domains with no slots
            lab_arr = [DUMMY_SLOT_LABEL]
        class_names = "#{0}".format(json.dumps(lab_arr))
        w_f.write(class_names + "\n")

        w_f.write(
            "\t".join(
                [
                    "#doc_index",
                    "doc_prediction",
                    "doc_label",
                    "doc_scores",
                    "word_predictions",
                    "word_labels",
                    "[word_pred:word_lab:score]",
                    "tokens",
                    "text",
                    "word_chunk_match",
                ]
            )
            + "\n"
        )

        doc_idx = 0
        for anno in annotations_list:

            if len(slot_labels) == 0:
                predictions = "0:0:" + DUMMY_SLOT_LABEL
                gold = predictions
            else:
                predictions = anno.predictions
                gold = anno.gold

            i_p_idx = intent_labels_list.index(anno.intent_pred)
            i_g_idx = intent_labels_list.index(anno.intent_gold)
            intent_scores = "[{}]".format(
                ",".join(
                    [
                        "'1'" if i == i_p_idx else "'0'"
                        for i in range(len(intent_labels_list))
                    ]
                )
            )
            pred_lab_scores = ""
            chunk_match = count_chunk_match(predictions, gold)
            doc_idx += 1
            w_f.write(
                "\t".join(
                    [
                        str(doc_idx),
                        str(i_p_idx),
                        str(i_g_idx),
                        intent_scores,
                        predictions,
                        gold,
                        pred_lab_scores,
                        str(anno.tokens),
                        anno.utterance.strip(),
                        chunk_match,
                    ]
                )
                + "\n"
            )
    print("Done writing output to {}".format(output_file))


def get_nonterminals(tree):
    return [x.label() for x in tree.subtrees()]


def parse_char_range_arguments(instr):
    labels = instr.split(",")
    for _lab in labels:
        start, end, _tag = _lab.split(":")
        yield int(start), int(end), _tag


def tree_to_predictions(tree):
    char_num = -1
    st = []
    slot_labels = set()

    intent_tree = tree.root.children[0]
    intent_label = remove_in_tag(intent_tree.label)

    if intent_tree:
        for n in intent_tree.children:
            if type(n) is Token:
                token = n.label
                char_num = char_num + len(token) + 1
            elif type(n) is Slot:
                # pre-terminal or slot label
                slot_label = remove_sl_tag(n.label)
                slot_labels.add(slot_label)
                num_char_start = char_num + 1
                for token in n.children:
                    char_num = char_num + len(token.label) + 1
                st.append(str(num_char_start) + ":" + str(char_num) + ":" + slot_label)
            else:
                raise ValueError("why is this here: " + str(type(n)))
    else:
        raise ValueError("Invalid Tree")

    return (",".join(st), slot_labels, intent_label)


def remove_sl_tag(l):
    return l.replace("SL:", "")


def remove_in_tag(l):
    return l.replace("IN:", "")


def count_chunk_match(predictions, labels):
    """
    for each prediction and label pair, count matched chunks
    return a json format string
    """
    chunk_match_dict = defaultdict(float)
    if predictions == "" or labels == "":
        return json.dumps(chunk_match_dict)

    def split_label(label):
        label = label.split(":")
        assert len(label) == 3, "label is " + str(label)
        tag = label[2]
        start = int(label[0])
        end = int(label[1])
        return (tag, start, end)

    label_list = [split_label(label) for label in labels.split(",")]
    for prediction in predictions.split(","):
        tag, start, end = split_label(prediction)
        for label in label_list:
            if (
                (abs(label[1] - start) <= 2)
                and (abs(label[2] - end) <= 2)
                and tag == label[0]
            ):
                chunk_match_dict[tag] += 1
                break
    return json.dumps(chunk_match_dict)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise "Arg1: gold file and Arg2: prediction file and Arg3: output file"
    gold_file = sys.argv[1]
    pred_file = sys.argv[2]
    output_file = sys.argv[3]
    prep_result_predictor(
        gold_file=gold_file, pred_file=pred_file, output_file=output_file
    )
