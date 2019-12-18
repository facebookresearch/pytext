#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import List

import torch
import torch.nn as nn
from pytext.optimizer.sparsifiers.sparsifier import L0_projection_sparsifier


class BlockwiseMagnitudeSparsifier(L0_projection_sparsifier):
    """
    running blockwise magnitude-based sparsification

    Args:
        block_size: define the size of each block

        columnwise_blocking: define columnwise block if true

        starting_epoch: sparsification_condition returns true only after starting_epoch

        frequency: sparsification_condition only if number of steps devides frequency

        accumulate_mask: if true, the mask after each .sparisfy() will be reused

        sparsity: percentage of zeros among the **UNPRUNED** parameters.


        Examples on how the sparsifier work:

        2D matrix:
        [
          0  1  2  3  4
          5  6  7  8  9
          10 11 12 13 14
          15 16 17 18 19
          20 21 22 23 24
        ]

        define 3 X 1 block
        [
          *********  *******
          *0  1  2*  *3   4*
          ********** *******
          *5  6  7*  *8   9*
          ********** *******
          *10 11 12* *13 14*
          ********** *******
          *15 16 17* *18 19*
          ********** *******
          *20 21 22* *23 24*
          ********** *******
        ]

        compute l1 norm of each block and sort them. Retain blocks with largest
        absolute values until sparsity threshold is met
    """

    class Config(L0_projection_sparsifier.Config):
        block_size: int = 16
        columnwise_blocking: bool = False
        accumulate_mask: bool = False
        layerwise_pruning: bool = True

    def __init__(
        self,
        sparsity,
        starting_epoch,
        frequency,
        block_size,
        columnwise_blocking,
        accumulate_mask,
        layerwise_pruning,
    ):
        super().__init__(sparsity, starting_epoch, frequency, layerwise_pruning)
        self.block_size = block_size
        self.columnwise_blocking = columnwise_blocking
        self.accumulate_mask = accumulate_mask
        self._masks = None
        assert self.layerwise_pruning, "layerwise pruning is forced"

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.sparsity,
            config.starting_epoch,
            config.frequency,
            config.block_size,
            config.columnwise_blocking,
            config.accumulate_mask,
            config.layerwise_pruning,
        )

    def get_sparsifiable_params(self, model, requires_name=False):
        sparsifiable_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and len(p.shape) == 2
        ]
        sparsifiable_params_name = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and len(p.shape) == 2
        ]
        if requires_name:
            return sparsifiable_params_name, sparsifiable_params
        else:
            return sparsifiable_params

    def get_current_sparsity(self, model):
        sparsifiable_params = self.get_sparsifiable_params(model)
        sparsifiable_params_count = sum(p.numel() for p in sparsifiable_params)
        nonzero_params = sum(p.nonzero().size(0) for p in sparsifiable_params)
        return (sparsifiable_params_count - nonzero_params) / sparsifiable_params_count

    def _padding_into_full_blocks(self, param):
        nrows, ncols = param.shape
        ncols_pad = math.ceil(ncols / self.block_size) * self.block_size
        padded_param = param.new_zeros((nrows, ncols_pad))
        padded_param[:nrows, :ncols] = param
        return padded_param

    def _num_blocks_kept(self, param, mask):
        if mask is None:
            mask = param.new_ones(param.shape)
        unpruned_param_sz = torch.nonzero(mask).size(0)
        max_num_nonzeros = math.ceil(unpruned_param_sz * (1 - self.sparsity))
        return math.ceil(max_num_nonzeros / self.block_size)

    def _compute_param_mask(
        self,
        param: torch.Tensor,
        pre_mask: torch.Tensor = None,
        columnwise_blocking: bool = False,
    ):
        if columnwise_blocking:
            return self._compute_param_mask(
                param.transpose(1, 0),
                pre_mask=(pre_mask.transpose(1, 0) if pre_mask else None),
            ).transpose(1, 0)
        padded_param = self._padding_into_full_blocks(param)
        if pre_mask is not None:
            padded_mask = self._padding_into_full_blocks(pre_mask)
            padded_param.data = padded_param.data * padded_mask

        block_l1norms = (
            torch.abs(padded_param).reshape(-1, 1, self.block_size).sum(dim=2)
        )
        max_num_blocks = self._num_blocks_kept(param, pre_mask)
        topk_threshold = (
            torch.topk(block_l1norms.flatten(), max_num_blocks).values.min().item()
        )
        mask = (
            block_l1norms.repeat(1, 1, self.block_size).reshape(padded_param.shape)
            >= topk_threshold
        ).to(param.dtype)
        if pre_mask is None:
            return mask[: param.size(0), : param.size(1)]
        else:
            return mask[: param.size(0), : param.size(1)] * pre_mask

    def get_masks(
        self, model: nn.Module, pre_masks: List[torch.Tensor] = None
    ) -> List[torch.Tensor]:

        learnableparams = self.get_sparsifiable_params(model)
        if pre_masks:
            self._masks = pre_masks

        if self._masks:
            assert len(learnableparams) == len(
                self._masks
            ), "parameter dimension and mask dimension does not match"
            for m, w in zip(self._masks, learnableparams):
                # check only for non-empty mask
                if len(m.size()):
                    assert (
                        m.size() == w.size()
                    ), "parameter dimension and mask dimension does not match"

        if self._masks is not None:
            # sparsifying 2D tensor only, skip mask for unlearnable
            # and unsparsifierable param
            masks = [
                self._compute_param_mask(p, m, self.columnwise_blocking)
                if len(p.shape) == 2 and p.requires_grad
                else p.new_empty(())
                for p, m in zip(learnableparams, self._masks)
            ]
        else:
            # sparsifying 2D tensor only, skip mask for unlearnable
            # and unsparsifierable param
            masks = [
                self._compute_param_mask(
                    p, columnwise_blocking=self.columnwise_blocking
                )
                if len(p.shape) == 2 and p.requires_grad
                else p.new_empty(())
                for p in learnableparams
            ]
        if self.accumulate_mask:
            self._masks = masks
        return masks
