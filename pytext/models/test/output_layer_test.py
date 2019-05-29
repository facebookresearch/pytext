#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import numpy as np
from pytext.data.tensorizers import LabelTensorizer
from pytext.data.utils import Vocabulary
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
