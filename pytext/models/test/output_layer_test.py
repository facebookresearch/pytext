#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import numpy as np
import torch
from pytext.data.tensorizers import LabelTensorizer
from pytext.data.utils import PAD, Vocabulary
from pytext.models.output_layers.lm_adaptive_softmax_output_layer import LMASOutputLayer
from pytext.models.output_layers.word_tagging_output_layer import WordTaggingOutputLayer


class OutputLayerTest(unittest.TestCase):
    def test_create_word_tagging_output_layer(self):
        tensorizer = LabelTensorizer()
        tensorizer.vocab = Vocabulary(["foo", "bar"])
        tensorizer.pad_idx = 0
        layer = WordTaggingOutputLayer.from_config(
            config=WordTaggingOutputLayer.Config(label_weights={"foo": 2.2}),
            labels=tensorizer.vocab,
        )
        np.testing.assert_array_almost_equal(
            np.array([2.2, 1]), layer.loss_fn.weight.detach().numpy()
        )

    def test_create_lm_adaptive_softmax_output_layer(self):
        tensorizer = LabelTensorizer()
        tensorizer.vocab = Vocabulary(
            ["lm", "lm1", "lm2", "lm3", "lm4", "lm5", "lm6", "lm7", "lm8", "lm9"]
        )
        tensorizer.vocab.idx[PAD] = -1
        layer = LMASOutputLayer.from_config(
            config=LMASOutputLayer.Config(cutoffs=[5]), labels=tensorizer.vocab
        )

        lmas = LMASOutputLayer(
            layer.in_features,
            layer.cutoffs,
            layer.target_names,
            layer.div_value,
            layer.head_bias,
            layer.pad_token_idx,
        )
        test_data = torch.randn(10, 1, 10)
        test_target = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        pred_test = torch.randn(2, 1, 10)
        assert lmas.get_loss(test_data, test_target, {"lmas": None}).item() > 0.0
        assert list(lmas.get_pred(pred_test)[0].size()) == [2, 1]
        assert list(lmas.get_pred(pred_test)[1].size()) == [2, 1, 10]
        assert (lmas.get_pred(pred_test)[1] < 0.0).all().item() == 1
