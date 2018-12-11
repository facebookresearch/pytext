#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Set, Union

from pytext.common.constants import BatchContext, DatasetFieldName, Stage
from pytext.data import CommonMetadata
from pytext.data.data_structures.annotation import (
    REDUCE,
    SHIFT,
    Intent,
    Slot,
    Token,
    Tree,
    TreeBuilder,
)
from pytext.metrics.intent_slot_metrics import (
    FramePredictionPair,
    Node,
    Span,
    compute_all_metrics,
)

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


PRED_TARGET_TREES = "pred_target_trees"


class CompositionalFileChannel(FileChannel):
    def get_title(self):
        return ("doc_index", "text", "predicted_annotation", "actual_annotation")

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        for index, utterance, (pred_tree, target_tree) in zip(
            context[BatchContext.INDEX],
            context[DatasetFieldName.UTTERANCE_FIELD],
            context[PRED_TARGET_TREES],
        ):
            yield (index, utterance, pred_tree.flat_str(), target_tree.flat_str())


class CompositionalMetricReporter(MetricReporter):
    def __init__(self, actions_vocab, channels: List[Channel]) -> None:
        super().__init__(channels)
        self.actions_vocab = actions_vocab

    @classmethod
    def from_config(cls, config, metadata: CommonMetadata):
        actions_vocab = metadata.actions_vocab.itos
        return cls(
            actions_vocab,
            [
                ConsoleChannel(),
                CompositionalFileChannel((Stage.TEST,), config.output_path),
            ],
        )

    def gen_extra_context(self):
        pred_target_trees = []
        for action_preds, action_targets, token_str_list in zip(
            self.all_preds, self.all_targets, self.all_context[DatasetFieldName.TOKENS]
        ):
            pred_tree = CompositionalMetricReporter.tree_from_tokens_and_indx_actions(
                token_str_list, self.actions_vocab, action_preds
            )
            target_tree = CompositionalMetricReporter.tree_from_tokens_and_indx_actions(
                token_str_list, self.actions_vocab, action_targets
            )
            pred_target_trees.append((pred_tree, target_tree))
        self.all_context[PRED_TARGET_TREES] = pred_target_trees

    def calculate_metric(self):
        return compute_all_metrics(
            [
                FramePredictionPair(
                    CompositionalMetricReporter.tree_to_metric_node(pred_tree),
                    CompositionalMetricReporter.tree_to_metric_node(target_tree),
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
        token_str_list: List[str], actions_vocab: List[str], actions_indices: List[int]
    ):
        builder = TreeBuilder()
        i = 0
        for action_idx in actions_indices:
            action = actions_vocab[action_idx]
            if action == REDUCE:
                builder.update_tree(action, None)
            elif action == SHIFT:
                builder.update_tree(action, token_str_list[i])
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
