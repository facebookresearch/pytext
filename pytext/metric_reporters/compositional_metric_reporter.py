#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Set, Union

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
from pytext.data.tensorizers import Tensorizer
from pytext.data.tokenizers import Tokenizer
from pytext.metrics.intent_slot_metrics import (
    FramePredictionPair,
    Node,
    Span,
    compute_all_metrics,
)

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


ALL_PRED_FRAMES = "all_pred_frames"


class CompositionalMetricReporter(MetricReporter):
    class Config(MetricReporter.Config):
        text_column_name: str = "tokenized_text"

    def __init__(
        self,
        actions_vocab,
        channels: List[Channel],
        text_column_name: str = Config.text_column_name,
        tokenizer: Tokenizer = None,
    ) -> None:
        super().__init__(channels)
        self.actions_vocab = actions_vocab
        self.text_column_name = text_column_name
        self.tokenizer = tokenizer or Tokenizer()
        self.pred_target_trees = None

    @classmethod
    def from_config(
        cls,
        config,
        metadata: CommonMetadata = None,
        tensorizers: Dict[str, Tensorizer] = None,
    ):
        if tensorizers is not None:
            return cls(
                tensorizers["actions"].vocab,
                [ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
                config.text_column_name,
                tensorizers["tokens"].tokenizer,
            )
        actions_vocab = metadata.actions_vocab.itos
        return cls(
            actions_vocab,
            [ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
        )

    def _reset(self):
        super()._reset()
        self.pred_target_trees = None

    def gen_extra_context(self):
        # check if all_preds contains top K results or only 1 result
        pred_target_trees = []
        all_pred_trees: List[List[Tree]] = []

        for top_k_action_preds, action_targets, token_str_list in zip(
            self.all_preds, self.all_targets, self.all_context[DatasetFieldName.TOKENS]
        ):
            topk_pred_trees = []
            for k, action_preds in enumerate(top_k_action_preds):
                pred_tree = CompositionalMetricReporter.tree_from_tokens_and_indx_actions(
                    token_str_list,
                    self.actions_vocab,
                    action_preds,
                    validate_tree=False,
                )
                topk_pred_trees.append(
                    CompositionalMetricReporter.tree_to_metric_node(pred_tree)
                )
                if k == 0:
                    target_tree = CompositionalMetricReporter.tree_from_tokens_and_indx_actions(
                        token_str_list, self.actions_vocab, action_targets
                    )
                    pred_target_trees.append((pred_tree, target_tree))
            all_pred_trees.append(topk_pred_trees)
        self.pred_target_trees = pred_target_trees
        self.all_context[ALL_PRED_FRAMES] = all_pred_trees

    def predictions_to_report(self):
        """
        Generate human readable predictions
        """
        return [t[0].flat_str() for t in self.pred_target_trees]

    def targets_to_report(self):
        """
        Generate human readable targets
        """
        return [t[1].flat_str() for t in self.pred_target_trees]

    # CREATE NODES
    def calculate_metric(self):
        return compute_all_metrics(
            self.create_frame_prediction_pairs(),
            overall_metrics=True,
            all_predicted_frames=self.all_context[ALL_PRED_FRAMES],
        )

    def create_frame_prediction_pairs(self):
        return [
            FramePredictionPair(
                CompositionalMetricReporter.tree_to_metric_node(pred_tree),
                CompositionalMetricReporter.tree_to_metric_node(target_tree),
            )
            for pred_tree, target_tree in self.pred_target_trees
        ]

    def get_model_select_metric(self, metrics):
        return metrics.frame_accuracy

    def batch_context(self, raw_batch, batch):
        context = super().batch_context(raw_batch, batch)
        context[DatasetFieldName.TOKENS] = [
            [
                token.value
                for token in self.tokenizer.tokenize(row[self.text_column_name])
            ]
            for row in raw_batch
        ]
        context[BatchContext.INDEX] = [1]
        context[self.text_column_name] = [
            row[self.text_column_name] for row in raw_batch
        ]

        return context

    @staticmethod
    def tree_from_tokens_and_indx_actions(
        token_str_list: List[str],
        actions_vocab: List[str],
        actions_indices: List[int],
        validate_tree: bool = True,
    ):
        builder = TreeBuilder()
        i = 0
        try:
            for action_idx in actions_indices:
                action = actions_vocab[action_idx]
                if action == REDUCE:
                    builder.update_tree(action, None)
                elif action == SHIFT:
                    builder.update_tree(action, token_str_list[i])
                    i += 1
                else:
                    builder.update_tree(action, action)

        except IndexError:
            builder = TreeBuilder()
            builder.update_tree(SHIFT, "IN:INVALID")
        tree = builder.finalize_tree(validate_tree=validate_tree)
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
        if node.children:
            for child in node.children:
                if type(child) == Token:
                    idx += len(child.label) + 1
                elif type(child) == Intent or type(child) == Slot:
                    res_child = CompositionalMetricReporter.node_to_metrics_node(
                        child, idx
                    )
                    res_children.add(res_child)
                    idx = res_child.span.end + 1
                else:
                    raise ValueError("Child must be Token, Intent or Slot!")
        node = Node(label=node.label, span=Span(start, idx - 1), children=res_children)
        return node
