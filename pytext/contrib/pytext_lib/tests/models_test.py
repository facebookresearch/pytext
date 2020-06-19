#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch.nn as nn
from pytext.contrib.pytext_lib import models


class TestModels(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_xlmr_dummy(self):
        model = models.xlm_roberta_dummy_binary_doc_classifier(pretrained=False)
        assert isinstance(model, nn.Module)
