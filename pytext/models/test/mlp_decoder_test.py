#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
import torch.nn as nn
from pytext.models.decoders.mlp_decoder import MLPDecoder


class MLPDecoderTest(unittest.TestCase):
    def _get_test_model(self, in_dim, out_dim, temp):
        class Model(nn.Module):
            def __init__(self, in_dim, out_dim, temp):
                super().__init__()

                self.mlp = MLPDecoder.from_config(
                    MLPDecoder.Config(bias=False, temperature=temp), in_dim, out_dim
                )

            def forward(self, tensor):
                return self.mlp(tensor)

        return Model(in_dim, out_dim, temp)

    def _tie_test_model_weights(self, model1, model2):
        mlp_weight = model2.mlp.mlp[0].weight.data
        model1.mlp.mlp[0].weight.data = mlp_weight

    def _check_logits_equality(self, logits1, logits2):
        return torch.isclose(logits1, logits2).flatten()

    def test_temperature(self):
        inputs = torch.rand(2, 2)

        model_with_temp = self._get_test_model(2, 2, 10)
        model_with_no_temp = self._get_test_model(2, 2, 1)
        self._tie_test_model_weights(model_with_temp, model_with_no_temp)

        # check logits are not temp scaled during training

        model_with_temp.train()
        model_with_no_temp.train()

        temp_logits = model_with_temp(inputs)
        no_temp_logits = model_with_no_temp(inputs)

        for model in (model_with_temp, model_with_no_temp):
            self.assertTrue(model.training)

        self.assertTrue(torch.equal(no_temp_logits, temp_logits))

        # check logits are temp scaled during testing

        model_with_temp.eval()
        model_with_no_temp.eval()

        temp_logits = model_with_temp(inputs)
        no_temp_logits = model_with_no_temp(inputs)

        for model in (model_with_temp, model_with_no_temp):
            self.assertFalse(model.training)

        self.assertFalse(any(self._check_logits_equality(no_temp_logits, temp_logits)))
        self.assertTrue(
            all(self._check_logits_equality(no_temp_logits, temp_logits * 10))
        )
