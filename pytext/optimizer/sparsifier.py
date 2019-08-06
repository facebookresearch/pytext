#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import List

import torch
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.models.model import Model


class Sparsifier(Component):
    __COMPONENT_TYPE__ = ComponentType.SPARSIFIER
    __EXPANSIBLE__ = True

    class Config(ConfigBase):
        pass

    def sparsify(self, *args, **kwargs):
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

    def __init__(self, sparsity, starting_epoch, frequency, layerwise_pruning=True):
        assert 0 <= sparsity <= 1
        self.sparsity = sparsity
        assert starting_epoch >= 1
        self.starting_epoch = starting_epoch
        assert frequency >= 1
        self.frequency = frequency
        self.layerwise_pruning = layerwise_pruning

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.sparsity,
            config.starting_epoch,
            config.frequency,
            config.layerwise_pruning,
        )

    def sparsify(self, model: Model):
        """
        obtain a mask and apply the mask to sparsify
        """
        print("running L0 projection-based (unstructured) sparsification. \n ")
        masks = self.get_masks(model)
        self.apply_masks(model, masks)

    def apply_masks(self, model: Model, masks: List[torch.Tensor]):
        """
        apply given masks to zero-out learnable weights in model
        """
        learnableparams = [p for p in model.parameters() if p.requires_grad]
        assert len(learnableparams) == len(masks)
        for m, w in zip(masks, learnableparams):
            assert m.size() == w.size()
            w.data *= m.clone()

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
        learnableparams = [p for p in model.parameters() if p.requires_grad]
        if pre_masks is None:
            # retain everything if no pre_masks given
            pre_masks = [torch.ones_like(p) for p in learnableparams]
        assert len(learnableparams) == len(pre_masks)
        for m, w in zip(pre_masks, learnableparams):
            assert m.size() == w.size()

        if self.layerwise_pruning:
            masks = []
            for m, param in zip(pre_masks, learnableparams):
                weights_abs = torch.abs(param.data).to(param.device)
                # absolute value of weights selected from pre_masks
                weights_abs_masked_flat = torch.flatten(weights_abs[m.bool()])
                total_size = weights_abs_masked_flat.numel()
                # using ceil instead of floor() or int()
                # because at least one element in the tensor required to be selected
                max_num_nonzeros = math.ceil(total_size * (1 - self.sparsity))
                # only pruned among the weights slected from pre_masks
                topkval = (
                    torch.topk(weights_abs_masked_flat, max_num_nonzeros)
                    .values.min()
                    .item()
                )
                # intersection of the new mask and pre_masks,
                # mask == 1 retain, mask == 0 pruned,
                mask = (weights_abs >= topkval).float() * m
                masks.append(mask)
        else:
            # concatenated flatten tensor of learnableparams that have pre_masks as True
            learnableparams_masked_flat = torch.cat(
                [
                    torch.flatten(p[m.bool()])
                    for m, p in zip(pre_masks, learnableparams)
                ],
                dim=0,
            )
            # using ceil instead of floor() or int() because at least one element
            # in the tensor required to be selected
            max_num_nonzeros = math.ceil(
                learnableparams_masked_flat.numel() * (1 - self.sparsity)
            )
            # select globally the top-k th weight among weights selected from pre_masks
            topkval = (
                torch.topk(torch.abs(learnableparams_masked_flat), max_num_nonzeros)
                .values.min()
                .item()
            )
            # intersection of the new mask and pre_masks,
            # mask == 1 retain, mask == 0 pruned,
            masks = [
                (torch.abs(p.data) >= topkval).float() * m
                for m, p in zip(pre_masks, learnableparams)
            ]

        return masks
