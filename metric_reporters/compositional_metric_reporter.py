#!/usr/bin/env python3

from typing import List, Set, Union

from pytext.common.constants import DatasetFieldName, Stage
from pytext.data import CommonMetadata
from pytext.fb.rnng.annotation import Intent, Slot, Token, Tree, TreeBuilder
from pytext.fb.rnng.utils import REDUCE, SHIFT, BiDict
from pytext.metrics import FramePredictionPair, Node, Span, compute_all_metrics

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


PRED_TARGET_TREES = "pred_target_trees"


class CompositionalFileChannel(FileChannel):
    def get_title(self):
        return ("doc_index", "text", "predicted_annotation", "actual_annotation")

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        for index, utterance, (pred_tree, target_tree) in zip(
            context[DatasetFieldName.INDEX_FIELD],
            context[DatasetFieldName.UTTERANCE_FIELD],
            context[PRED_TARGET_TREES],
        ):
            yield (index, utterance, pred_tree.flat_str(), target_tree.flat_str())


class CompositionalMetricReporter(MetricReporter):
    model_select_metric_name = "frame_accuracy"

    def __init__(self, actions_bidict: BiDict, channels: List[Channel]) -> None:
        super().__init__(channels)
        # This will be removed and done via data handler context in D10101440.
        self.actions_bidict = actions_bidict

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        actions_bidict = meta.actions_bidict
        return cls(
            actions_bidict,
            [
                ConsoleChannel(),
                CompositionalFileChannel((Stage.TEST), config.output_path),
            ],
        )

    def gen_extra_context(self):
        self.all_context[PRED_TARGET_TREES] = [
            (
                self.tree_from_tokens_and_indx_actions(
                    token_range[0], self.actions_bidict, action_preds
                ),
                self.tree_from_tokens_and_indx_actions(
                    token_range[0], self.actions_bidict, action_targets
                ),
            )
            for action_preds, action_targets, token_range in zip(
                self.all_preds,
                self.all_targets,
                self.all_context[DatasetFieldName.TOKEN_RANGE],
            )
        ]

    def calculate_metric(self):
        return compute_all_metrics(
            [
                FramePredictionPair(
                    self.tree_to_metric_node(pred_tree),
                    self.tree_to_metric_node(target_tree),
                )
                for pred_tree, target_tree in self.all_context[PRED_TARGET_TREES]
            ],
            overall_metrics=True,
        )

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.frame_accuracy

    @staticmethod
    def tree_from_tokens_and_indx_actions(
        tokens_str, actions_dict: BiDict, actions_indices
    ):
        builder = TreeBuilder()
        i = 0
        for action_idx in actions_indices:
            action = actions_dict.value(action_idx)
            if action == REDUCE:
                builder.update_tree(action, None)
            elif action == SHIFT:
                builder.update_tree(action, tokens_str[i])
                i += 1
            else:
                builder.update_tree(action, action)
        tree = builder.finalize_tree()
        return tree

    @staticmethod
    def tree_to_metric_node(tree: Tree) -> Node:
        """
        Creates a Node from tree assuming the utterance is a concatenation of the
        tokens by whitespaces. The function does not necessarily reproduce the original
        utterance as extra whitespaces can be introduced.
        """
        return CompositionalMetricReporter.node_to_metrics_node(tree.root.children[0])

    @staticmethod
    def node_to_metrics_node(node: Union[Intent, Slot], start: int = 0) -> Node:
        """
        The input start is the absolute start position in utterance
        """
        res_children: Set[Node] = set()
        idx = start
        for child in node.children:
            if type(child) == Token:
                idx += len(child.label) + 1
            elif type(child) == Intent or type(child) == Slot:
                res_child = CompositionalMetricReporter.node_to_metrics_node(child, idx)
                res_children.add(res_child)
                idx = res_child.span.end + 1
            else:
                raise ValueError("Child must be Token, Intent or Slot!")
        node = Node(label=node.label, span=Span(start, idx - 1), children=res_children)
        return node
