#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import unittest

import torch
from pytext.optimizer.sparsifiers.blockwise_sparsifier import (
    BlockwiseMagnitudeSparsifier,
)


class TestSparsifier(unittest.TestCase):
    def _get_blockwise_sparsifier(
        self, block_size, sparsity, columnwise_blocking=False
    ):
        config = BlockwiseMagnitudeSparsifier.Config(
            block_size=block_size,
            sparsity=sparsity,
            columnwise_blocking=columnwise_blocking,
        )
        return BlockwiseMagnitudeSparsifier.from_config(config)

    def test_param_padding(self):
        blockwise_sparsifier = self._get_blockwise_sparsifier(2, 0.6)
        param = torch.tensor([i for i in range(25)], dtype=float).view(5, 5)
        true_padded_param = param.new_zeros(param.size(0), 6)
        true_padded_param[:, : param.size(1)] = param
        padded_param = blockwise_sparsifier._padding_into_full_blocks(param)
        self.assertTrue(
            torch.allclose(
                torch.sum(true_padded_param - padded_param),
                torch.tensor([0.0], dtype=float),
            )
        )

    def test_param_mask(self):
        sparsity = 0.6
        block_sz = 3
        blockwise_sparsifier = self._get_blockwise_sparsifier(block_sz, sparsity)
        param = torch.tensor([i for i in range(100)], dtype=float).view(10, 10)
        param = param - param.new_ones(param.shape) * 50
        block_l1_norms = param.new_zeros(param.shape)
        # loop-based implementation to compute blockwise l1norm
        abs_vals = []
        for i in range(10):
            for j in range(0, 10, block_sz):
                block_l1_norms[i, j : j + block_sz] = torch.sum(
                    torch.abs(param[i, j : j + block_sz])
                )
                abs_vals.append(torch.sum(torch.abs(param[i, j : j + block_sz])))

        nnz = math.ceil(100 * (1 - sparsity))
        nnz_blocks = math.ceil(nnz / block_sz)
        abs_vals.sort(reverse=True)
        threshold = abs_vals[nnz_blocks - 1]
        true_mask = (block_l1_norms >= threshold).to(param.dtype)
        mask = blockwise_sparsifier._compute_param_mask(param)
        diff = torch.sum(torch.abs(mask - true_mask)).item()
        self.assertTrue(diff <= 0.00000001)

    def test_param_mask_columnwise(self):
        sparsity = 0.6
        block_sz = 3
        blockwise_sparsifier = self._get_blockwise_sparsifier(
            block_sz, sparsity, columnwise_blocking=True
        )
        param = torch.tensor([i for i in range(100)], dtype=float).view(10, 10)
        block_l1_norms = param.new_zeros(param.shape)
        # loop-based implementation to compute blockwise l1norm
        abs_vals = []
        for i in range(10):
            for j in range(0, 10, block_sz):
                block_l1_norms[j : j + block_sz, i] = torch.sum(
                    torch.abs(param[j : j + block_sz, i])
                )
                abs_vals.append(torch.sum(torch.abs(param[j : j + block_sz, i])))

        nnz = math.ceil(100 * (1 - sparsity))
        nnz_blocks = math.ceil(nnz / block_sz)
        abs_vals.sort(reverse=True)
        threshold = abs_vals[nnz_blocks - 1]
        true_mask = (block_l1_norms >= threshold).to(param.dtype)
        mask = blockwise_sparsifier._compute_param_mask(param, columnwise_blocking=True)
        diff = torch.sum(torch.abs(mask - true_mask)).item()
        self.assertTrue(diff <= 0.00000001)

    def test_param_mask_with_pre_mask(self):
        sparsity = 0.6
        block_sz = 3
        blockwise_sparsifier = self._get_blockwise_sparsifier(block_sz, sparsity)
        param = torch.tensor([i for i in range(100)], dtype=float).view(10, 10)
        param = param - param.new_ones(param.shape) * 50
        pre_mask = param.new_zeros(param.shape)
        pre_mask[5:, :] = torch.ones(pre_mask[5:, :].shape)
        block_l1_norms = param.new_zeros(param.shape)
        # loop-based implementation to compute blockwise l1norm
        abs_vals = []
        for i in range(10):
            for j in range(0, 10, block_sz):
                block_l1_norms[i, j : j + block_sz] = torch.sum(
                    torch.abs(param[i, j : j + block_sz])
                    * pre_mask[i, j : j + block_sz]
                )
                abs_vals.append(
                    torch.sum(
                        torch.abs(param[i, j : j + block_sz])
                        * pre_mask[i, j : j + block_sz]
                    )
                )

        nnz_unpruned = torch.nonzero(pre_mask).size(0)
        max_num_nonzeros = math.ceil(nnz_unpruned * (1 - sparsity))
        nnz_blocks = math.ceil(max_num_nonzeros / block_sz)
        abs_vals.sort(reverse=True)
        threshold = abs_vals[nnz_blocks - 1]
        true_mask = (block_l1_norms >= threshold).to(param.dtype)
        mask = blockwise_sparsifier._compute_param_mask(param, pre_mask)
        diff = torch.sum(torch.abs(mask - true_mask)).item()
        self.assertTrue(diff <= 0.00000001)
