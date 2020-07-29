#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
from pytext.contrib.pytext_lib import models
from pytext.contrib.pytext_lib.trainers import CompatibleTrainer, SimpleTrainer
from pytext.optimizer import Adam
from torch.utils.data import DataLoader


class TestTrainers(unittest.TestCase):
    def setUp(self):
        self.TORCH_NUM_THREAD = torch.get_num_threads()
        # workaround to get rid of Segmentation Fault in unit test
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.TORCH_NUM_THREAD)

    def test_simple_trainer(self):
        train_dataloader = self._generate_data()
        model = models.xlmr_dummy_binary_doc_classifier(pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.00001, eps=1e-8)
        trainer = SimpleTrainer()
        trained_model = trainer.fit(train_dataloader, model, optimizer, epoch=1)
        assert isinstance(trained_model, nn.Module)

    def test_compatible_trainer(self):
        train_dataloader = self._generate_data()
        val_dataloader = self._generate_data()
        model = models.xlmr_dummy_binary_doc_classifier(pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.00001, eps=1e-8)
        trainer = CompatibleTrainer(model, epochs=1)
        trained_model, _ = trainer.train(
            train_dataloader, val_dataloader, model, optimizer, label_names=["0", "1"]
        )
        assert isinstance(trained_model, nn.Module)

    def _generate_data(self):
        dummy_dataset = [
            {
                "label_ids": torch.tensor([1, 0]),
                "positions": torch.tensor([[0, 1, 2], [0, 1, 2]]),
                "segment_labels": torch.tensor([[0, 0, 0], [0, 0, 0]]),
                "pad_mask": torch.tensor([[1, 1, 1]]),
                "token_ids": torch.tensor([[5, 8, 13], [3, 6, 9]]),
            }
        ]
        return DataLoader(dummy_dataset, batch_size=None)
