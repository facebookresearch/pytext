#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
from typing import Dict, List

import numpy as np
from pytext.common.constants import Stage
from pytext.data.tensorizers import Tensorizer
from pytext.metric_reporters.channel import ConsoleChannel
from pytext.metric_reporters.compositional_metric_reporter import (
    CompositionalMetricReporter,
)

# These classes have been migrated to the open source directories. Imported
# here for compatibility purposes.
from pytext.metric_reporters.seq2seq_compositional import (  # noqa
    CompositionalSeq2SeqFileChannel,
    Seq2SeqCompositionalMetricReporter,
)
from pytext.metric_reporters.seq2seq_utils import stringify
from pytext.metrics.intent_slot_metrics import FramePredictionPair
from pytext.metrics.mask_metrics import (
    compute_masked_metrics,
    compute_nas_masked_metrics,
)

from .compositional_utils import extract_beam_subtrees, filter_invalid_beams


class MaskedCompositionalSeq2SeqFileChannel(CompositionalSeq2SeqFileChannel):
    SAMPLEWISE_LOSSES = (
        # logging losses
        "samplewise_label_loss",
        "samplewise_length_loss",
        "samplewise_labels_cross_entropy_loss",
        "samplewise_labels_label_smoothing_loss",
        "samplewise_lengths_cross_entropy_loss",
        "samplewise_lengths_label_smoothing_loss",
    )

    def get_title(self, context_keys=()):
        original_titles = super().get_title(context_keys=context_keys)
        new_titles = (
            "predicted_length",
            "target_length",
            "topk_tree_predictions",
            "topk_raw_predictions",
            "prediction_score",
            "topk_prediction_scores",
        )
        return (
            original_titles
            + new_titles
            + MaskedCompositionalSeq2SeqFileChannel.SAMPLEWISE_LOSSES
        )

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        for i, row in enumerate(
            super().gen_content(metrics, loss, preds, targets, scores, context)
        ):
            topk_raw_preds = [
                self.tensorizers["trg_seq_tokens"].stringify(pred).upper()
                for pred in context["topk_tree_predictions"][i]
            ]
            topk_preds = [self.validated_annotation(pred) for pred in topk_raw_preds]
            samplewise_losses = [
                (
                    context["losses"][loss_name][i]
                    if ("losses" in context and loss_name in context["losses"])
                    else None
                )
                for loss_name in MaskedCompositionalSeq2SeqFileChannel.SAMPLEWISE_LOSSES
            ]
            row.extend(
                [
                    context["predicted_length"][i],
                    context["target_length"][i],
                    topk_preds,
                    topk_raw_preds,
                    # scores is assumed to be in descending order
                    #    so we individually log the highest score to sort the output file later
                    scores[i][0],
                    scores[i],
                ]
                + samplewise_losses
            )

            yield row


class MaskedSeq2SeqCompositionalMetricReporter(Seq2SeqCompositionalMetricReporter):
    class Config(Seq2SeqCompositionalMetricReporter.Config):
        model_select_metric_key: str = "all_loss"
        select_length_beam: int = 0
        log_gradient: bool = True
        TEMP_DUMP_PREDICTIONS: bool = True
        log_samplewise_losses: bool = True

    def __init__(
        self,
        channels,
        log_gradient,
        tensorizers,
        accept_flat_intents_slots,
        model_select_metric_key,
        select_length_beam,
    ):
        super().__init__(channels, log_gradient, tensorizers, accept_flat_intents_slots)
        self.model_select_metric_key = model_select_metric_key
        if model_select_metric_key == "fa":
            self.lower_is_better = False
        else:
            self.lower_is_better = True
        self.select_length_beam = select_length_beam

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        channels = [ConsoleChannel()]
        if config.TEMP_DUMP_PREDICTIONS:
            channels.append(
                MaskedCompositionalSeq2SeqFileChannel(
                    [Stage.TEST],
                    config.output_path,
                    tensorizers,
                    config.accept_flat_intents_slots,
                )
            )
        return cls(
            channels,
            config.log_gradient,
            tensorizers,
            config.accept_flat_intents_slots,
            config.model_select_metric_key,
            config.select_length_beam,
        )

    def _reset(self):
        super()._reset()
        self.all_target_length_preds: List = []
        self.all_beam_preds: List[List[str]] = []
        self.all_top_non_invalid: List[str] = []
        self.all_top_extract: List[str] = []
        self.all_loss: Dict[str, List] = defaultdict(list)

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        super().add_batch_stats(
            n_batches, preds, targets, scores, None, m_input, **context
        )
        self.all_loss["all_loss"].append(float(loss[0]))

        custom_losses = loss[1].keys()
        for loss_name in custom_losses:
            vals = self.all_loss[loss_name]
            # samplewise losses are stored as multi-element tensors, so need to separate cases
            if "samplewise" in loss_name:
                vals.append(loss[1][loss_name].data.cpu().numpy())
            else:
                vals.append(float(loss[1][loss_name]))

    def calculate_loss(self):
        """
        Calculate the average loss for all aggregated batch
        """
        loss_agg = {}
        for loss_name in self.all_loss.keys():
            if "samplewise" in loss_name:
                self.all_context.setdefault("losses", {})[loss_name] = np.concatenate(
                    self.all_loss[loss_name], axis=None
                )
            else:
                loss_agg[loss_name] = np.average(
                    self.all_loss[loss_name], weights=self.batch_size
                )

        return loss_agg

    def calculate_metric(self):
        all_metrics = compute_masked_metrics(
            self.create_frame_prediction_pairs(
                self.all_pred_trees, self.all_target_trees
            ),
            self.all_target_lens,
            self.all_target_length_preds,
            self.select_length_beam,
            overall_metrics=True,
            calculated_loss=self.calculate_loss(),
            all_predicted_frames=self.all_beam_preds,
            non_invalid_frame_pairs=self.create_frame_prediction_pairs(
                self.all_top_non_invalid, self.all_target_trees
            ),
            extracted_frame_pairs=self.create_frame_prediction_pairs(
                self.all_top_extract, self.all_target_trees
            ),
        )
        return all_metrics

    def create_frame_prediction_pairs(self, all_pred_trees, all_target_trees):
        return [
            FramePredictionPair(
                CompositionalMetricReporter.tree_to_metric_node(pred_tree),
                CompositionalMetricReporter.tree_to_metric_node(target_tree),
            )
            for pred_tree, target_tree in zip(all_pred_trees, all_target_trees)
        ]

    def get_top_non_invalid(self, beam: List[str]) -> str:
        beam_tokens: List[List[str]] = list(map(lambda pred: pred.split(), beam))
        non_invalid = filter_invalid_beams(beam_tokens)
        if len(non_invalid) > 0:
            return " ".join(non_invalid[0])
        else:
            return beam[0]

    def get_top_extract(self, beam: List[str]) -> str:
        beam_tokens: List[List[str]] = list(map(lambda pred: pred.split(), beam))
        subtrees = extract_beam_subtrees(beam_tokens)
        if len(subtrees) > 0:
            return " ".join(subtrees[0])
        else:
            return beam[0]

    def aggregate_preds(self, new_batch, context=None):
        if new_batch is None:
            return
        tree_preds = new_batch[0]  # bsz X beam_size X seq_len
        length_preds = new_batch[1]
        target_vocab = self.tensorizers["trg_seq_tokens"].vocab
        target_pad_token = target_vocab.get_pad_index()
        target_bos_token = target_vocab.get_bos_index()
        target_eos_token = target_vocab.get_eos_index()

        cleaned_preds = [
            self._remove_tokens(
                pred, [target_pad_token, target_eos_token, target_bos_token]
            )
            for pred in self._make_simple_list(tree_preds)
        ]
        self.aggregate_data(self.all_preds, cleaned_preds)

        pred_trees = [
            self.stringify_annotation_tree(pred[0], target_vocab)
            for pred in cleaned_preds
        ]

        beam_pred_trees = [
            [
                CompositionalMetricReporter.tree_to_metric_node(
                    self.stringify_annotation_tree(pred, target_vocab)
                )
                for pred in beam
            ]
            for beam in cleaned_preds
        ]

        top_non_invalid_trees = [
            self.get_annotation_from_string(
                self.get_top_non_invalid(
                    [stringify(pred, target_vocab) for pred in beam]
                )
            )
            for beam in cleaned_preds
        ]

        top_extracted_trees = [
            self.get_annotation_from_string(
                self.get_top_extract([stringify(pred, target_vocab) for pred in beam])
            )
            for beam in cleaned_preds
        ]

        self.aggregate_data(self.all_pred_trees, pred_trees)
        self.aggregate_data(self.all_target_length_preds, length_preds)
        self.aggregate_data(self.all_beam_preds, beam_pred_trees)
        self.aggregate_data(self.all_top_non_invalid, top_non_invalid_trees)
        self.aggregate_data(self.all_top_extract, top_extracted_trees)

    def get_model_select_metric(self, metrics):
        if self.model_select_metric_key == "fa":
            return metrics.frame_accuracy
        else:
            return metrics.loss[self.model_select_metric_key]

    def gen_extra_context(self, *args):
        super().gen_extra_context()
        self.all_context["predicted_length"] = self.all_target_length_preds
        self.all_context["target_length"] = self.all_target_lens
        self.all_context["topk_tree_predictions"] = self.all_preds


class NASMaskedSeq2SeqCompositionalMetricReporter(
    MaskedSeq2SeqCompositionalMetricReporter
):
    class Config(MaskedSeq2SeqCompositionalMetricReporter.Config):
        ref_frame_accuracy: float = 1.0
        ref_model_num_param: float = 1.0
        param_importance: float = 1.0

    def __init__(
        self,
        channels,
        log_gradient,
        tensorizers,
        accept_flat_intents_slots,
        model_select_metric_key,
        select_length_beam,
        ref_frame_accuracy,
        ref_model_num_param,
        param_importance,
    ):
        super().__init__(
            channels,
            log_gradient,
            tensorizers,
            accept_flat_intents_slots,
            model_select_metric_key,
            select_length_beam,
        )

        self.ref_frame_accuracy = ref_frame_accuracy
        self.ref_model_num_param = ref_model_num_param
        self.param_importance = param_importance

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        channels = [ConsoleChannel()]
        if config.TEMP_DUMP_PREDICTIONS:
            channels.append(
                MaskedCompositionalSeq2SeqFileChannel(
                    [Stage.TEST],
                    config.output_path,
                    tensorizers,
                    config.accept_flat_intents_slots,
                )
            )
        return cls(
            channels,
            config.log_gradient,
            tensorizers,
            config.accept_flat_intents_slots,
            config.model_select_metric_key,
            config.select_length_beam,
            config.ref_frame_accuracy,
            config.ref_model_num_param,
            config.param_importance,
        )

    def gen_extra_context(self, model):
        super().gen_extra_context()
        assert (
            model is not None
        ), "Must include a model for total number of parameter calculation"
        self.total_param_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    def calculate_metric(self):
        all_metrics = compute_nas_masked_metrics(
            self.create_frame_prediction_pairs(
                self.all_pred_trees, self.all_target_trees
            ),
            self.all_target_lens,
            self.all_target_length_preds,
            self.select_length_beam,
            overall_metrics=True,
            calculated_loss=self.calculate_loss(),
            all_predicted_frames=self.all_beam_preds,
            non_invalid_frame_pairs=self.create_frame_prediction_pairs(
                self.all_top_non_invalid, self.all_target_trees
            ),
            extracted_frame_pairs=self.create_frame_prediction_pairs(
                self.all_top_extract, self.all_target_trees
            ),
            model_num_param=self.total_param_count,
            ref_model_num_param=self.ref_model_num_param,
            ref_frame_accuracy=self.ref_frame_accuracy,
            param_importance=self.param_importance,
        )
        return all_metrics
