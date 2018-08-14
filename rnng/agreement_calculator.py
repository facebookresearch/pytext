#!/usr/bin/env python3

from collections import defaultdict, namedtuple
from pytext.rnng.annotation import Slot
import copy
from itertools import combinations
import numpy as np
from pandas import DataFrame
from pytext.rnng.utils import INTENT_PREFIX
from typing import Counter, DefaultDict

DEFAULTLABEL = "noLabel"
P_R_F1 = namedtuple(
    "P_R_F1",
    [
        "precision",
        "recall",
        "f1",
        "num_gold",
        "num_pred",
        "num_correct",
        "top_confusions",
    ],
)


class Evaluation:
    def __init__(self):
        self.per_label_metrics = defaultdict()  # type: dict[str, P_R_F1]
        self.metrics = defaultdict()
        self.all_scores = None  # type: P_R_F1

    def print_eval(self):
        print("\nPrecision/Recall/F1 scores for labels")
        for label, label_scores in self.per_label_metrics.items():
            print(
                "Label: {},  P={:.2f} ({}/{}), R={:.2f} ({}/{}), F1={:.2f};\
                 Top confusions: {}".format(  # noqa: B950
                    label,
                    label_scores.precision * 100,
                    label_scores.num_correct,
                    label_scores.num_pred,
                    label_scores.recall * 100,
                    label_scores.num_correct,
                    label_scores.num_gold,
                    label_scores.f1 * 100,
                    label_scores.top_confusions,
                )
            )
        print(
            "Total: Precision {:.2f} ({}/{}), Recall: {:.2f} ({}/{}), \
            F1: {:.2f}".format(
                self.all_scores.precision * 100,
                self.all_scores.num_correct,
                self.all_scores.num_pred,
                self.all_scores.recall * 100,
                self.all_scores.num_correct,
                self.all_scores.num_gold,
                self.all_scores.f1 * 100,
            )
        )


class ConfusionMatrix:
    def __init__(self):
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

    def update(self, gold, pred):
        self.confusion_matrix[gold][pred] += 1

    def compute_metrics(
        self,
        note_for_metrics="",
        filter_out_noLabel=True,
        filter_out_intentlabels=False,
    ) -> Evaluation:
        evaluation = Evaluation()

        print("\n*Metrics for " + str(note_for_metrics) + "*")
        df = DataFrame(self.confusion_matrix).T.fillna(0)
        all_labels = set(df.index.union(df.columns))

        if filter_out_intentlabels:
            valid_labels = np.array(
                [x for x in all_labels if not x.startswith(INTENT_PREFIX)]
            )
        else:
            valid_labels = np.array(list(all_labels))

        num_classes = len(valid_labels)

        df = df.reindex(index=valid_labels, columns=valid_labels).fillna(0)

        self.confusion_matrix = df.values

        assert self.confusion_matrix.shape[0] == self.confusion_matrix.shape[1]
        assert self.confusion_matrix.shape[0] == num_classes

        prediction_counter: Counter[str] = Counter()
        correct_counter = DefaultDict[str, float]()

        precision_class = np.zeros(num_classes)
        recall_class = np.zeros(num_classes)
        f1_class = np.zeros(num_classes)
        reference_instances = np.zeros(num_classes)

        # Confusion matrix has gold labels down the row index and pred labels
        # down the column index

        intent_indices = np.zeros(num_classes)

        num_classes_for_avg = 0
        for i, label in enumerate(valid_labels):
            if label == DEFAULTLABEL:
                continue

            num_classes_for_avg += 1
            if label.startswith(INTENT_PREFIX):
                intent_indices[i] = 1

            reference_instances[i] = sum(self.confusion_matrix[i, :])
            prediction_counter[label] = sum(self.confusion_matrix[:, i])
            correct_counter[label] = self.confusion_matrix[i, i]

            recall_class[i] = (
                self.confusion_matrix[i, i] / reference_instances[i]
                if reference_instances[i] > 0
                else 0
            )

            precision_class[i] = (
                self.confusion_matrix[i, i] / prediction_counter[label]
                if prediction_counter[label] > 0
                else 0
            )

            if precision_class[i] + recall_class[i] > 0:
                f1_class[i] = (
                    2
                    * precision_class[i]
                    * recall_class[i]
                    / (precision_class[i] + recall_class[i])
                )
            else:
                f1_class[i] = 0

            top_confusions = get_top_confusions(
                i, self.confusion_matrix[i, :], valid_labels
            )
            evaluation.per_label_metrics[label] = P_R_F1(
                num_correct=correct_counter[label],
                precision=precision_class[i],
                recall=recall_class[i],
                f1=f1_class[i],
                top_confusions=top_confusions,
                num_gold=reference_instances[i],
                num_pred=prediction_counter[label],
            )

        all_correct = sum(correct_counter.values())
        predicted_num = sum(prediction_counter.values())
        gold_num = sum(reference_instances)

        precision_all = (
            sum(correct_counter.values()) / sum(prediction_counter.values())
            if sum(prediction_counter.values())
            else 0
        )
        recall_all = (
            sum(correct_counter.values()) / sum(reference_instances)
            if sum(reference_instances)
            else 0
        )
        f1_all = (
            2 * precision_all * recall_all / (precision_all + recall_all)
            if precision_all and recall_all
            else 0
        )

        evaluation.all_scores = P_R_F1(
            precision=precision_all,
            recall=recall_all,
            f1=f1_all,
            top_confusions=None,
            num_gold=gold_num,
            num_pred=predicted_num,
            num_correct=all_correct,
        )
        return evaluation


def get_top_confusions(index, confusions, valid_labels):
    top_confusion_indices = np.argsort(confusions[confusions > 0])[-3:]
    top_confusion_indices = top_confusion_indices[top_confusion_indices != index][::-1]
    return "; ".join(
        [
            str(x)
            for x in zip(
                valid_labels[top_confusion_indices], confusions[top_confusion_indices]
            )
        ]
    )


class Calculator:
    """
    This is a generic class for calculating inter-annotator agreement
    Classes inheiriting from this one should overwirte:
        self.name
        self.tree_similarity
    """

    def __init__(self, deletion=False):
        self.name = "Generic Agreement"
        self.deletion = deletion

    def tree_similarity(self, instance, confusion_matrix=None):
        print("Comparison not implemented")
        return 0.0

    @staticmethod
    def delete_matching_non_slot_tokens(tree_1, tree_2):
        copy_1 = copy.deepcopy(tree_1)
        copy_2 = copy.deepcopy(tree_2)

        # Store terminals by index
        terminals_1 = {t.index: t for t in copy_1.root.list_terminals()}
        terminals_2 = {t.index: t for t in copy_2.root.list_terminals()}

        # Find terminals to remove
        indices_1 = {
            i for i in terminals_1.keys() if type(terminals_1[i].parent) != Slot
        }
        indices_2 = [
            i for i in terminals_2.keys() if type(terminals_2[i].parent) != Slot
        ]
        indices_to_remove = indices_1.intersection(indices_2)

        for i in indices_to_remove:
            terminals_1[i].remove()
            terminals_2[i].remove()

        return copy_1, copy_2

    @staticmethod
    def delete_non_slot_tokens(parse):
        parse_copy = copy.deepcopy(parse)

        terminals = parse_copy.root.list_terminals()
        for t in terminals:
            if type(t.parent) != Slot:
                # Disconnect the terminal from the tree
                t.remove()
        return parse_copy


class Label_and_Span_Calculator(Calculator):
    def __init__(self, deletion):
        super(Label_and_Span_Calculator, self).__init__(deletion)
        self.name = "Partial Agreement: Label and Span matching"
        # If tokens are deleted, save instances where the score changed

    def tree_similarity(self, instance, confusion_matrix=None):
        """
        If token deletion is indicated, return the score with token deletion
        but calculate it both ways, and track sentences with differences
        """
        gold_tree = instance[0]
        pred_tree = instance[1]
        if not self.deletion:
            return self._similarity(gold_tree, pred_tree, confusion_matrix)
        if self.deletion:
            # Replace the parses with deep copies
            # Some terminal nodes have been deleted
            #   tokens falling outside Slots in both parses
            copy_1, copy_2 = Calculator.delete_matching_non_slot_tokens(
                gold_tree, pred_tree
            )
            new_metric = self._similarity(copy_1, copy_2)
            return new_metric

    def _similarity(self, tree_1, tree_2, confusion_matrix=None):
        nodes1 = tree_1.root.list_nonTerminals()
        nodes2 = tree_2.root.list_nonTerminals()

        node_info_1 = set()
        for node in nodes1:
            node_info_1.add((node.label, node.get_token_span()))

        node_info_2 = set()
        for node in nodes2:
            node_info_2.add((node.label, node.get_token_span()))

        matching = node_info_1.intersection(node_info_2)
        """
        precision = len(matching) / len(nodes1)
        recall = len(matching) / len(nodes2)
        if precision == 0 and recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / ( precision + recall )
        """
        # Simplification of the equations above
        f1 = 2 * len(matching) / (len(nodes1) + len(nodes2))

        if confusion_matrix is not None:
            self._compare_spans(node_info_1, node_info_2, confusion_matrix)
        return f1

    def _compare_spans(self, node_info_1, node_info_2, confusion_matrix):

        for (l_gold, span_gold) in node_info_1:
            matched_span = False
            for (l_pred, span_pred) in node_info_2:
                if span_gold == span_pred:
                    confusion_matrix.update(gold=l_gold, pred=l_pred)
                    matched_span = True
                    break
            if not matched_span:
                confusion_matrix.update(gold=l_gold, pred=DEFAULTLABEL)

        for (l_pred, span_pred) in node_info_2:
            matched_span = False
            for (_, span_gold) in node_info_1:
                if span_gold == span_pred:
                    matched_span = True
                    break
            if not matched_span:
                confusion_matrix.update(gold=DEFAULTLABEL, pred=l_pred)


class Strict_Label_and_Span_Calculator(Label_and_Span_Calculator):
    """ A node is correct if the subtree at that node in the gold tree
        matches exactly with the subtree at the node in the pred tree
    """

    def __init__(self):
        super(Label_and_Span_Calculator, self).__init__(False)

        self.name = "Strict Label and Span matching"

    def _similarity(self, gold_tree, pred_tree, confusion_matrix=None):
        nodes1 = gold_tree.root.list_nonTerminals()
        nodes2 = pred_tree.root.list_nonTerminals()

        node_info_1 = set()
        for node in nodes1:
            node_info_1.add((node.label, node.children_flat_str_spans()))

        node_info_2 = set()
        for node in nodes2:
            node_info_2.add((node.label, node.children_flat_str_spans()))

        matching = node_info_1.intersection(node_info_2)
        """
        precision = len(matching) / len(nodes1)
        recall = len(matching) / len(nodes2)
        if precision == 0 and recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / ( precision + recall )
        """
        # Simplification of the equations above
        f1 = 2 * len(matching) / (len(nodes1) + len(nodes2))

        if confusion_matrix is not None:
            self._compare_spans(node_info_1, node_info_2, confusion_matrix)
        return f1


class Total_Agreement_Calculator(Calculator):
    def __init__(self, deletion):
        super(Total_Agreement_Calculator, self).__init__(deletion)
        self.name = "Total Agreement"

    def tree_similarity(self, instance):
        tree_1 = instance[0]
        tree_2 = instance[1]

        if self.deletion:
            tree_1 = Calculator.delete_non_slot_tokens(tree_1)
            tree_2 = Calculator.delete_non_slot_tokens(tree_2)

        if tree_1 == tree_2:
            return 1.0
        else:
            return 0.0


class ThreeWay_Agreement_Calculator(Calculator):
    """
    For Each Triplet of Annotators, find the percent of times 2/3 agree
    Return the average of the triplet scores.
    Obviously, this requires at least 3 annotators
    """

    def __init__(self, deletion):
        super(ThreeWay_Agreement_Calculator, self).__init__(deletion)
        self.name = "3-Way Agreement: 2/3 Total Match"

    def tree_similarity(self, instance):
        """
        Return 1 if two or more of the annotators agree (out of 3)
        Else 0
        """

        if self.deletion:
            instance = [Calculator.delete_non_slot_tokens(i) for i in instance]

        for pair in combinations(instance, 2):
            if pair[0] == pair[1]:
                return 1
        return 0


if __name__ == "__main__":
    print("Don't call this")
