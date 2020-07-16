#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import json
import math
import os
import sys
from enum import Enum
from typing import List

import numpy as np
import torch
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.models.crf import CRF
from pytext.models.model import Model
from pytext.utils import timing
from pytext.utils.file_io import PathManager


class State(Enum):
    ANALYSIS = "Analysis"
    OTHERS = "Others"


class Sparsifier(Component):
    __COMPONENT_TYPE__ = ComponentType.SPARSIFIER
    __EXPANSIBLE__ = True

    class Config(ConfigBase):
        pass

    def sparsify(self, *args, **kwargs):
        pass

    def sparsification_condition(self, *args, **kwargs):
        pass

    def get_sparsifiable_params(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        pass

    def get_current_sparsity(self, model: Model) -> float:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        nonzero_params = sum(
            p.nonzero().size(0) for p in model.parameters() if p.requires_grad
        )
        return (trainable_params - nonzero_params) / trainable_params


class L0_projection_sparsifier(Sparsifier):
    """
    L0 projection-based (unstructured) sparsification

    Args:
        weights (torch.Tensor): input weight matrix
        sparsity (float32): the desired sparsity [0-1]

    """

    class Config(Sparsifier.Config):
        sparsity: float = 0.9
        starting_epoch: int = 2
        frequency: int = 1
        layerwise_pruning: bool = True
        accumulate_mask: bool = False

    def __init__(
        self,
        sparsity,
        starting_epoch,
        frequency,
        layerwise_pruning=True,
        accumulate_mask=False,
    ):
        assert 0 <= sparsity <= 1
        self.sparsity = sparsity
        assert starting_epoch >= 1
        self.starting_epoch = starting_epoch
        assert frequency >= 1
        self.frequency = frequency
        self.layerwise_pruning = layerwise_pruning
        self.accumulate_mask = accumulate_mask
        self._masks = None

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.sparsity,
            config.starting_epoch,
            config.frequency,
            config.layerwise_pruning,
            config.accumulate_mask,
        )

    def sparsification_condition(self, state):
        return (
            state.stage == Stage.TRAIN
            and state.epoch >= self.starting_epoch
            and state.step_counter % self.frequency == 0
        )

    def sparsify(self, state):
        """
        obtain a mask and apply the mask to sparsify
        """
        model = state.model
        # compute new mask when conditions are True
        if self.sparsification_condition(state):
            masks = self.get_masks(model)
            # applied the computed mask, self.accumulate_mask handled separately
            if not self.accumulate_mask:
                self.apply_masks(model, masks)

        # if self.accumulate_mask is True, apply the existent mask irregardless Stage
        if self.accumulate_mask and self._masks is not None:
            self.apply_masks(model, self._masks)

    def get_sparsifiable_params(self, model: Model):
        sparsifiable_params = [p for p in model.parameters() if p.requires_grad]
        return sparsifiable_params

    def apply_masks(self, model: Model, masks: List[torch.Tensor]):
        """
        apply given masks to zero-out learnable weights in model
        """
        learnableparams = self.get_sparsifiable_params(model)
        assert len(learnableparams) == len(masks)
        for m, w in zip(masks, learnableparams):
            if len(m.size()):
                assert m.size() == w.size()
                w.data *= m.clone()
                # if accumulate_mask, remove a param permanently by also removing
                # its gradient
                if self.accumulate_mask:
                    w.grad.data *= m.clone()

    def get_masks(
        self, model: Model, pre_masks: List[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Note: this function returns the masks only but do not sparsify or modify the
        weights

        prune x% of weights among the weights with "1" in pre_masks

        Args:
            model: Model
            pre_masks: list of FloatTensors where "1" means retained the weight and
             "0" means pruned the weight

        Return:
            masks: List[torch.Tensor], intersection of new masks and pre_masks, so
            that "1" only if the weight is selected after new masking and pre_mask
        """
        learnableparams = self.get_sparsifiable_params(model)
        if pre_masks:
            self._masks = pre_masks
        if self._masks is None:
            # retain everything if no pre_masks given
            self._masks = [torch.ones_like(p) for p in learnableparams]

        assert len(learnableparams) == len(self._masks)
        for m, w in zip(self._masks, learnableparams):
            if len(m.size()):
                assert m.size() == w.size()

        if self.layerwise_pruning:
            masks = []
            for m, param in zip(self._masks, learnableparams):
                weights_abs = torch.abs(param.data).to(param.device)
                # absolute value of weights selected from existent masks
                weights_abs_masked_flat = torch.flatten(weights_abs[m.bool()])
                total_size = weights_abs_masked_flat.numel()
                if total_size > 0:
                    # using ceil instead of floor() or int()
                    # because at least one element in the tensor required to be selected
                    max_num_nonzeros = math.ceil(total_size * (1 - self.sparsity))
                    # only pruned among the weights slected from existent masks
                    topkval = (
                        torch.topk(weights_abs_masked_flat, max_num_nonzeros)
                        .values.min()
                        .item()
                    )
                    # intersection of the new mask and pre_mexistent masks,
                    # mask == 1 retain, mask == 0 pruned,
                    mask = (weights_abs >= topkval).float() * m
                else:
                    mask = param.new_empty(())
                masks.append(mask)
        else:
            # concatenated flatten tensor of learnableparams that have _masks as True
            learnableparams_masked_flat = torch.cat(
                [
                    torch.flatten(p[m.bool()])
                    for m, p in zip(self._masks, learnableparams)
                ],
                dim=0,
            )
            # using ceil instead of floor() or int() because at least one element
            # in the tensor required to be selected
            max_num_nonzeros = math.ceil(
                learnableparams_masked_flat.numel() * (1 - self.sparsity)
            )
            # select globally the top-k th weight among weights selected from _masks
            topkval = (
                torch.topk(torch.abs(learnableparams_masked_flat), max_num_nonzeros)
                .values.min()
                .item()
            )
            # intersection of the new mask and _masks,
            # mask == 1 retain, mask == 0 pruned,
            masks = [
                (torch.abs(p.data) >= topkval).float() * m
                if p.numel() > 0
                else p.new_empty(())
                for m, p in zip(self._masks, learnableparams)
            ]

        if self.accumulate_mask:
            self._masks = masks

        return masks


class CRF_SparsifierBase(Sparsifier):
    class Config(Sparsifier.Config):
        starting_epoch: int = 1
        frequency: int = 1

    def sparsification_condition(self, state):
        if state.stage == Stage.TRAIN:
            return False

        return (
            state.epoch >= self.starting_epoch
            and state.step_counter % self.frequency == 0
        )

    def get_sparsifiable_params(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, CRF):
                return m.transitions.data

    def get_transition_sparsity(self, transition):
        nonzero_params = transition.nonzero().size(0)
        return (transition.numel() - nonzero_params) / transition.numel()


class CRF_L1_SoftThresholding(CRF_SparsifierBase):
    """
    implement l1 regularization:
        min Loss(x, y, CRFparams) + lambda_l1 * ||CRFparams||_1

    and solve the optimiation problem via (stochastic) proximal gradient-based
    method i.e., soft-thresholding

    param_updated = sign(CRFparams) * max ( abs(CRFparams) - lambda_l1, 0)
    """

    class Config(CRF_SparsifierBase.Config):
        lambda_l1: float = 0.001

    def __init__(self, lambda_l1: float, starting_epoch: int, frequency: int):
        self.lambda_l1 = lambda_l1
        assert starting_epoch >= 1
        self.starting_epoch = starting_epoch
        assert frequency >= 1
        self.frequency = frequency

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.lambda_l1, config.starting_epoch, config.frequency)

    def sparsify(self, state):
        if not self.sparsification_condition(state):
            return
        model = state.model
        transition_matrix = self.get_sparsifiable_params(model)
        transition_matrix_abs = torch.abs(transition_matrix)
        assert (
            len(state.optimizer.param_groups) == 1
        ), "different learning rates for multiple param groups not supported"
        lrs = state.optimizer.param_groups[0]["lr"]
        threshold = self.lambda_l1 * lrs
        transition_matrix = torch.sign(transition_matrix) * torch.max(
            (transition_matrix_abs - threshold),
            transition_matrix.new_zeros(transition_matrix.shape),
        )
        current_sparsity = self.get_transition_sparsity(transition_matrix)
        print(f"sparsity of CRF transition matrix: {current_sparsity}")


class CRF_MagnitudeThresholding(CRF_SparsifierBase):
    """
    magnitude-based (equivalent to projection onto l0 constraint set) sparsification
    on CRF transition matrix. Preserveing the top-k elements either rowwise or
    columnwise until sparsity constraint is met.
    """

    class Config(CRF_SparsifierBase.Config):
        sparsity: float = 0.9
        grouping: str = "row"

    def __init__(self, sparsity, starting_epoch, frequency, grouping):
        assert 0 <= sparsity <= 1
        self.sparsity = sparsity
        assert starting_epoch >= 1
        self.starting_epoch = starting_epoch
        assert frequency >= 1
        self.frequency = frequency
        assert (
            grouping == "row" or grouping == "column"
        ), "grouping needs to be row or column"
        self.grouping = grouping

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.sparsity, config.starting_epoch, config.frequency, config.grouping
        )

    def sparsify(self, state):
        if not self.sparsification_condition(state):
            return
        model = state.model
        transition_matrix = self.get_sparsifiable_params(model)
        num_rows, num_cols = transition_matrix.shape
        trans_abs = torch.abs(transition_matrix)
        if self.grouping == "row":
            max_num_nonzeros = math.ceil(num_cols * (1 - self.sparsity))
            topkvals = (
                torch.topk(trans_abs, k=max_num_nonzeros, dim=1)
                .values.min(dim=1, keepdim=True)
                .values
            )

        else:
            max_num_nonzeros = math.ceil(num_rows * (1 - self.sparsity))
            topkvals = (
                torch.topk(trans_abs, k=max_num_nonzeros, dim=0)
                .values.min(dim=0, keepdim=True)
                .values
            )

        # trans_abs < topkvals is a broadcasted comparison
        transition_matrix[trans_abs < topkvals] = 0.0
        current_sparsity = self.get_transition_sparsity(transition_matrix)
        print(f"sparsity of CRF transition matrix: {current_sparsity}")


class SensitivityAnalysisSparsifier(Sparsifier):
    class Config(Sparsifier.Config):
        pre_train_model_path: str = ""
        analyzed_sparsity: float = 0.8
        # we don't use all eval data for analysis, only use a portion of the data.
        max_analysis_batches: int = 0
        # allow the user to skip pruning for some weight. Here we set the max
        # number of weight tensor can be skipped for pruning.
        max_skipped_weight: int = 0
        # if we already did sensitivity analysis before
        pre_analysis_path: str = ""
        sparsity: float = 0.8

    def __init__(
        self,
        pre_train_model_path,
        analyzed_sparsity,
        max_analysis_batches,
        max_skipped_weight,
        pre_analysis_path,
        sparsity,
    ):
        assert PathManager.exists(
            pre_train_model_path
        ), "The pre-trained model must be exist"
        self.pre_train_model_path = pre_train_model_path
        self.param_dict = None
        assert (
            0.0 <= analyzed_sparsity <= 1.0
        ), "Analyzed sparsity need to be in the range of [0, 1]"
        self.analyzed_sparsity = analyzed_sparsity
        self.max_analysis_batches = max_analysis_batches
        self.max_skipped_weight = max_skipped_weight
        self.require_mask_parameters = []
        self.pre_analysis_path = pre_analysis_path
        assert (
            0.0 <= sparsity <= 1.0
        ), "Pruning sparsity need to be in the range of [0, 1]"
        self.sparsity = sparsity
        self._masks = None
        self.analysis_state = State.OTHERS

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.pre_train_model_path,
            config.analyzed_sparsity,
            config.max_analysis_batches,
            config.max_skipped_weight,
            config.pre_analysis_path,
            config.sparsity,
        )

    def get_sparsifiable_params(self, model):
        param_dict = {}
        for module_name, m in model.named_modules():
            # Search the name of all module_name in named_modules
            # only test the parameters in nn.Linear
            if isinstance(m, nn.Linear):
                # module_name: module.xxx
                # param_name: module.xxx.weight
                # we only check weight tensor
                param_name = module_name + ".weight"
                param_dict[param_name] = m.weight

        return param_dict

    def get_mask_for_param(self, param, sparsity):
        """
        generate the prune mask for one weight tensor.
        """
        n = int(sparsity * param.nelement())
        if n > 0:
            # If n > 0, we need to remove n parameters, the threshold
            # equals to the n-th largest parameters.x
            threshold = float(param.abs().flatten().kthvalue(n - 1)[0])
        else:
            # If n == 0, it means all parameters need to be kept.
            # Because the absolute parameter value >= 0, setting
            # threshold to -1 ensures param.abs().ge(threshold)
            # is True for all the parameters.
            threshold = -1.0
        # reverse_mask indiciates the weights that need to be kept
        mask = param.abs().ge(threshold).float()

        return mask

    def layer_wise_analysis(
        self, param_name, param_dict, trainer, state, eval_data, metric_reporter
    ):
        # perform pruning for the target param with param_name
        if param_name is None:
            prunable_param_shape = None
        else:
            prunable_param = param_dict[param_name]
            # include the shape information for better analysis
            prunable_param_shape = list(prunable_param.shape)
            mask = self.get_mask_for_param(prunable_param, self.analyzed_sparsity)
            with torch.no_grad():
                param_dict[param_name].data.mul_(mask)
        # get the eval_metric for the pruned model
        with torch.no_grad():
            # set the number of batches of eval data for analysis
            analysis_data = eval_data
            if self.max_analysis_batches > 0:
                analysis_data = itertools.islice(eval_data, self.max_analysis_batches)
            eval_metric = trainer.run_epoch(state, analysis_data, metric_reporter)
        current_metric = metric_reporter.get_model_select_metric(eval_metric)
        if metric_reporter.lower_is_better:
            current_metric = -current_metric

        return current_metric, prunable_param_shape

    def find_params_to_prune(self, metric_dict, max_skip_weight_num):
        require_mask_parameters = sorted(
            metric_dict.keys(), reverse=True, key=lambda param: metric_dict[param]
        )
        metric_sensitivities_by_param = [
            metric_dict[p] for p in require_mask_parameters
        ]

        skipped_weight_num = 0
        while skipped_weight_num < max_skip_weight_num:
            # calculate the mean and sandard deviation
            mean_ = np.mean(metric_sensitivities_by_param[:-skipped_weight_num])
            std_ = np.std(metric_sensitivities_by_param[:-skipped_weight_num])
            # skip runing of the parameter if the metric disensitivity is
            # less than mean_ - 3 * std_, otherwise break.
            if (
                metric_sensitivities_by_param[-skipped_weight_num - 1]
                >= mean_ - 3 * std_
            ):
                break
            skipped_weight_num += 1

        require_mask_parameters = require_mask_parameters[:-skipped_weight_num]

        # return how many weight are skipped during this iteration
        return require_mask_parameters, skipped_weight_num

    def sensitivity_analysis(
        self, trainer, state, eval_data, metric_reporter, train_config
    ):
        """
        Analysis the sensitivity of each weight tensor to the metric.
        Prune the weight tensor one by one and evaluate the metric if the
        correspond weight tensor is pruned.
        Args:
            trainer (trainer): batch iterator of training data
            state (TrainingState): the state of the current training
            eval_data (BatchIterator): batch iterator of evaluation data
            metric_reporter (MetricReporter): compute metric based on training
            output and report results to console, file.. etc
            train_config (PyTextConfig): training config

        Returns:
            analysis_result: a string of each layer sensitivity to metric.
        """
        print("Analyzed_sparsity: {}".format(self.analyzed_sparsity))
        print("Evaluation metric_reporter: {}".format(type(metric_reporter).__name__))
        output_path = (
            os.path.dirname(train_config.task.metric_reporter.output_path)
            + "/sensitivity_analysis_sparsifier.ckp"
        )

        # param_dict: the dict maps weight tensor to the parameter name
        self.param_dict = self.get_sparsifiable_params(state.model)

        # load the pretrained model
        print("load the pretrained model from: " + self.pre_train_model_path)
        self.loaded_model = torch.load(
            self.pre_train_model_path, map_location=torch.device("cpu")
        )

        # set model to evaluation mode
        state.stage = Stage.EVAL
        state.model.eval(Stage.EVAL)

        metric_dict = {}
        all_param_list = [None] + list(self.param_dict.keys())
        print("All prunable parameters", all_param_list)

        # print the sensitivity results for each weight
        print("#" * 40)
        print("save the analysis result to: ", output_path)
        print("Pruning Sensitivity Test: param / shape / eval metric")

        # iterate through all_param_list to test pruning snesitivity
        for param_name in all_param_list:
            print("=" * 40)
            print("Testing {}".format(param_name))
            state.model.load_state_dict(self.loaded_model["model_state"])

            current_metric, prunable_param_shape = self.layer_wise_analysis(
                param_name, self.param_dict, trainer, state, eval_data, metric_reporter
            )
            if param_name is None:
                baseline_metric = current_metric
            metric_dict[param_name] = current_metric - baseline_metric
        print("#" * 40)

        # remove baseline metric from the analysis results
        if None in metric_dict:
            del metric_dict[None]
        # write the test result into the checkpoint
        if state.rank == 0:
            with PathManager.open(output_path, "w") as fp:
                json.dump(metric_dict, fp)

        return metric_dict

    def sparsification_condition(self, state):
        return state.stage == Stage.TRAIN

    def apply_masks(self, model: Model, masks: List[torch.Tensor]):
        """
        apply given masks to zero-out learnable weights in model
        """
        learnable_params = self.get_required_sparsifiable_params(model)
        assert len(learnable_params) == len(masks)
        for m, w in zip(masks, learnable_params):
            if len(m.size()):
                assert m.size() == w.size()
                w.data *= m

    def get_current_sparsity(self, model: Model) -> float:
        trainable_params = sum(
            module.weight.data.numel()
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        )
        nonzero_params = sum(
            module.weight.data.nonzero().size(0)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        )
        return (trainable_params - nonzero_params) / trainable_params

    def sparsify(self, state):
        """
        apply the mask to sparsify the weight tensor
        """
        # do not sparsify the weight tensor during the analysis
        if self.analysis_state == State.ANALYSIS:
            return

        model = state.model
        # compute new mask when conditions are True
        if self.sparsification_condition(state):
            # applied the computed mask to sparsify the weight
            self.apply_masks(model, self._masks)

    def get_required_sparsifiable_params(self, model: Model):
        # param_dict contains all parameters, select requied weights
        # if we reload analysis result from file, we need to calculate
        # all param_dict again.
        if self.param_dict is None:
            self.param_dict = self.get_sparsifiable_params(model)

        return [self.param_dict[p] for p in self.require_mask_parameters]

    def get_masks(self, model: Model) -> List[torch.Tensor]:
        """
        Note: this function returns the masks for each weight tensor if
        that tensor is required to be pruned

        prune x% of weights items among the weights with "1" in mask (self._mask)
        indicate the remained weights, with "0" indicate pruned weights

        Args:
            model: Model

        Return:
            masks: List[torch.Tensor], the prune mask for the weight of all
            layers
        """
        learnable_params = self.get_required_sparsifiable_params(model)

        masks = []
        for param in learnable_params:
            mask = self.get_mask_for_param(param, self.sparsity)
            masks.append(mask)

        return masks

    def load_analysis_from_path(self):
        assert PathManager.isfile(self.pre_analysis_path), "{} is not a file".format(
            self.pre_analysis_path
        )
        with PathManager.open(self.pre_analysis_path, "r") as fp:
            metric_dict = json.load(fp)

        return metric_dict

    @timing.time("sparsifier initialize")
    def initialize(self, trainer, state, eval_data, metric_reporter, train_config):
        # if user specify the analysis file, load it from path
        if self.pre_analysis_path:
            metric_dict = self.load_analysis_from_path()

        else:
            self.analysis_state = State.ANALYSIS
            metric_dict = self.sensitivity_analysis(
                trainer, state, eval_data, metric_reporter, train_config
            )
            # finish the analysis, sparsifier can apply prune mask.
            self.analysis_state = State.OTHERS

        # skip some of the weight tensors from pruning. The user can
        # specify the max_skipped_weight, which limit the max number
        # of weight to be skipped.
        self.require_mask_parameters, skipped_weight_num = self.find_params_to_prune(
            metric_dict, self.max_skipped_weight
        )

        for p in self.require_mask_parameters:
            print(p, " ", metric_dict[p])
        print("#" * 40)
        sys.stdout.flush()
        print(str(skipped_weight_num) + " weight tensors are skipped for pruning")

        # initialize and generate the pruning mask. We don't want to generate
        # the mask for each step. Otherwise, it will be time inefficient.
        self._masks = self.get_masks(state.model)
