#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import List

import torch
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.models.crf import CRF
from pytext.models.model import Model


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
