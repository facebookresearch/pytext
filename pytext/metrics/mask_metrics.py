#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, NamedTuple, Optional, Sequence

from pytext.metrics import (
    ClassificationMetrics,
    LabelPrediction,
    compute_classification_metrics,
)
from pytext.metrics.intent_slot_metrics import (
    FrameAccuraciesByDepth,
    FramePredictionPair,
    IntentSlotMetrics,
    Node,
    compute_all_metrics,
    compute_frame_accuracy,
)
from sklearn.metrics import accuracy_score

BIT2BYTE_CONSTANT = 2 ** 23


class MaskedSeq2SeqJointMetrics(NamedTuple):
    top_intent_accuracy: Optional[float]
    current_model_parameter_size: Optional[float]
    size_32bit_model: Optional[float]
    size_8bit_model: Optional[float]
    size_4bit_model: Optional[float]
    frame_accuracy: Optional[float]
    frame_accuracy_top_k: Optional[float]
    frame_accuracies_by_depth: Optional[FrameAccuraciesByDepth]
    bracket_metrics: Optional[IntentSlotMetrics]
    tree_metrics: Optional[IntentSlotMetrics]
    percent_invalid_trees: Optional[float]
    percent_trees_wrong_label: Optional[float]
    loss: Optional[float] = None
    length_metrics: Dict[int, float] = None
    length_reports: ClassificationMetrics = None
    non_invalid_fa: float = 0.0
    extracted_fa: float = 0.0
    print_length_metrics: bool = True

    def print_metrics(self) -> None:
        if self.size_32bit_model:
            print(f"\n\nsize of 32bit model = {self.size_32bit_model} MB")
        if self.size_8bit_model:
            print(f"\n\nsize of 8bit model = {self.size_8bit_model} MB")
        if self.size_4bit_model:
            print(f"\n\nsize of 4bit model = {self.size_4bit_model} MB")
        if self.current_model_parameter_size:
            print(
                f"\n\nparameter size of current model = {self.current_model_parameter_size} MB"
            )
        if self.frame_accuracy:
            print(f"\n\nFrame accuracy = {self.frame_accuracy * 100:.2f}")
        if self.frame_accuracy_top_k:
            print(f"\n\nTop k frame accuracy = {self.frame_accuracy_top_k * 100:.2f}")
        if self.bracket_metrics:
            print("\n\nBracket Metrics")
            self.bracket_metrics.print_metrics()
        if self.tree_metrics:
            print("\n\nTree Metrics")
            self.tree_metrics.print_metrics()
        if self.length_metrics:
            print("\n\nLength Metrics :", self.length_metrics)
            print(f"Length Accuracy: {self.length_reports.accuracy * 100:.2f}")
        if self.length_reports and self.print_length_metrics:
            print("\n\nLength Reports :", self.length_reports.print_metrics())
        if self.non_invalid_fa:
            print(f"Non Invalid FA {self.non_invalid_fa}")
        if self.extracted_fa:
            print(f"Extracted FA {self.extracted_fa}")
        if self.percent_invalid_trees is not None:
            print(f"\n\n% Invalid Trees = {self.percent_invalid_trees * 100:.2f}%")
        if self.percent_trees_wrong_label is not None:
            print(
                f"\n\n% Trees w/ wrong IN/SL = {self.percent_trees_wrong_label * 100:.2f}%"
            )


class NASMaskedSeq2SeqJointMetrics(MaskedSeq2SeqJointMetrics):
    def __new__(cls, **kwargs):
        model_num_param = kwargs.pop("model_num_param")
        norm_prod_model_param_frame_accuracy = kwargs.pop(
            "norm_prod_model_param_frame_accuracy"
        )
        self = super(NASMaskedSeq2SeqJointMetrics, cls).__new__(cls, **kwargs)
        self.model_num_param = model_num_param
        self.norm_prod_model_param_frame_accuracy = norm_prod_model_param_frame_accuracy
        return self

    def print_metrics(self) -> None:
        super(NASMaskedSeq2SeqJointMetrics, self).print_metrics()

        if self.model_num_param:
            print(f"\nNumber of Parameters {self.model_num_param}")
        if self.norm_prod_model_param_frame_accuracy is not None:
            print(
                f"Normalized product of model parameter and frame accuracy {self.norm_prod_model_param_frame_accuracy}"
            )


def compute_length_metrics(
    all_target_lens: List[int],
    all_target_length_preds: List[List[int]],
    select_length_beam,
    log_per_label_metrics: bool = True,
):
    length_metrics = {}
    length_report = {}
    if all_target_length_preds:
        all_length_pred_agg = {}
        beam = len(all_target_length_preds[0])
        for i in range(beam):
            all_length_pred_agg[i] = []
        for label, preds in zip(all_target_lens, all_target_length_preds):
            for l in range(beam):
                if label in preds[0 : l + 1]:
                    all_length_pred_agg[l].append(label)
                else:
                    all_length_pred_agg[l].append(preds[0])
        for i in range(beam):
            length_metrics[i] = accuracy_score(all_target_lens, all_length_pred_agg[i])

        max_len = max(all_target_lens + all_length_pred_agg[select_length_beam])
        all_pairs = [
            LabelPrediction(
                [1 if idx == pred else 0 for idx in range(max_len + 1)], pred, expect
            )
            for pred, expect in zip(
                all_length_pred_agg[select_length_beam], all_target_lens
            )
        ]

        length_report = compute_classification_metrics(
            all_pairs,
            [str(l) for l in range(max_len + 1)],
            0.0,  # Placeholder loss
            log_per_label_metrics=log_per_label_metrics,
        )

    return length_metrics, length_report


def compute_masked_metrics(
    frame_pairs: Sequence[FramePredictionPair],
    all_target_lens: List[int],
    all_target_length_preds: List[List[int]],
    select_length_beam: int,
    top_intent_accuracy: bool = True,
    frame_accuracy: bool = True,
    frame_accuracies_by_depth: bool = True,
    bracket_metrics: bool = True,
    tree_metrics: bool = True,
    overall_metrics: bool = False,
    all_predicted_frames: List[List[Node]] = None,
    calculated_loss: float = None,
    length_metrics: Dict = None,
    non_invalid_frame_pairs: Optional[Sequence[FramePredictionPair]] = None,
    extracted_frame_pairs: Optional[Sequence[FramePredictionPair]] = None,
    print_length_metrics: bool = True,
    num_weights=None,
    current_model_parameter_size=None,
) -> MaskedSeq2SeqJointMetrics:

    all_metrics = compute_all_metrics(
        frame_pairs,
        top_intent_accuracy,
        frame_accuracy,
        frame_accuracies_by_depth,
        bracket_metrics,
        tree_metrics,
        overall_metrics,
        all_predicted_frames,
        calculated_loss,
    )
    length_metrics, length_reports = compute_length_metrics(
        all_target_lens,
        all_target_length_preds,
        select_length_beam,
        log_per_label_metrics=print_length_metrics,
    )
    # Compute sizes of 32bit, 8bit, and 4bit versions of the model
    # in MB (hence the conversion factor), if the number of weights is provided.
    # Default to -1 if unavailable.
    size_32bit_model = -1
    size_8bit_model = -1
    size_4bit_model = -1
    if num_weights:
        size_32bit_model = 32 * num_weights / (BIT2BYTE_CONSTANT)
        size_8bit_model = 8 * num_weights / (BIT2BYTE_CONSTANT)
        size_4bit_model = 4 * num_weights / (BIT2BYTE_CONSTANT)
    if not current_model_parameter_size:
        current_model_parameter_size = -1

    return MaskedSeq2SeqJointMetrics(
        top_intent_accuracy=all_metrics.top_intent_accuracy,
        current_model_parameter_size=current_model_parameter_size,
        size_32bit_model=size_32bit_model,
        size_8bit_model=size_8bit_model,
        size_4bit_model=size_4bit_model,
        frame_accuracy=all_metrics.frame_accuracy,
        frame_accuracy_top_k=all_metrics.frame_accuracy_top_k,
        frame_accuracies_by_depth=all_metrics.frame_accuracies_by_depth,
        bracket_metrics=all_metrics.bracket_metrics,
        tree_metrics=all_metrics.tree_metrics,
        percent_invalid_trees=all_metrics.percent_invalid_trees,
        percent_trees_wrong_label=all_metrics.percent_trees_wrong_label,
        loss=all_metrics.loss,
        length_metrics=length_metrics,
        length_reports=length_reports,
        non_invalid_fa=compute_frame_accuracy(non_invalid_frame_pairs),
        extracted_fa=compute_frame_accuracy(extracted_frame_pairs),
        print_length_metrics=print_length_metrics,
    )


def compute_nas_masked_metrics(
    frame_pairs: Sequence[FramePredictionPair],
    all_target_lens: List[int],
    all_target_length_preds: List[List[int]],
    select_length_beam: int,
    top_intent_accuracy: bool = True,
    frame_accuracy: bool = True,
    frame_accuracies_by_depth: bool = True,
    bracket_metrics: bool = True,
    tree_metrics: bool = True,
    overall_metrics: bool = False,
    all_predicted_frames: List[List[Node]] = None,
    calculated_loss: float = None,
    length_metrics: Dict = None,
    non_invalid_frame_pairs: Optional[Sequence[FramePredictionPair]] = None,
    extracted_frame_pairs: Optional[Sequence[FramePredictionPair]] = None,
    model_num_param: float = 1.0,
    ref_model_num_param: float = 1.0,
    ref_frame_accuracy: float = 1.0,
    param_importance: float = 1.0,
) -> NASMaskedSeq2SeqJointMetrics:

    masked_seq2seq_joint_metrics = compute_masked_metrics(
        frame_pairs,
        all_target_lens,
        all_target_length_preds,
        select_length_beam,
        overall_metrics=True,
        calculated_loss=calculated_loss,
        all_predicted_frames=all_predicted_frames,
        non_invalid_frame_pairs=non_invalid_frame_pairs,
        extracted_frame_pairs=extracted_frame_pairs,
    )

    # Compute the objective function, which is defined as
    # (current_model_accuracy / ref_model_accuracy) *
    # ((# of parameters in a reference model) / (# of parameters in the current model)) ** alpha
    #  where alpha is a hyper-parameter to determine the improtance of the second term
    norm_prod_model_param_frame_accuracy = (
        masked_seq2seq_joint_metrics.frame_accuracy
        * ref_model_num_param ** param_importance
    ) / (ref_frame_accuracy * model_num_param ** param_importance)

    return NASMaskedSeq2SeqJointMetrics(
        top_intent_accuracy=masked_seq2seq_joint_metrics.top_intent_accuracy,
        frame_accuracy=masked_seq2seq_joint_metrics.frame_accuracy,
        frame_accuracy_top_k=masked_seq2seq_joint_metrics.frame_accuracy_top_k,
        frame_accuracies_by_depth=masked_seq2seq_joint_metrics.frame_accuracies_by_depth,
        bracket_metrics=masked_seq2seq_joint_metrics.bracket_metrics,
        tree_metrics=masked_seq2seq_joint_metrics.tree_metrics,
        loss=masked_seq2seq_joint_metrics.loss,
        length_metrics=masked_seq2seq_joint_metrics.length_metrics,
        length_reports=masked_seq2seq_joint_metrics.length_reports,
        non_invalid_fa=masked_seq2seq_joint_metrics.non_invalid_fa,
        extracted_fa=masked_seq2seq_joint_metrics.extracted_fa,
        model_num_param=model_num_param,
        norm_prod_model_param_frame_accuracy=norm_prod_model_param_frame_accuracy,
    )
