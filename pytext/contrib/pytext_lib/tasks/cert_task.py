#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from copy import deepcopy

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.contrib.pytext_lib.transforms import ModelTransform


logger = logging.getLogger(__name__)


# Most of the logic comes from <https://github.com/facebookresearch/moco>.
# I've adapted this to Lightning by adding the required methods, and implementing an extra optimiser.
class Cert(pl.LightningModule):
    def __init__(
        self,
        model,
        optim,
        datamodule,
        transform: ModelTransform,
        embedding_dim: int,
        queue_size: int,
        temperature: float = 0.07,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.batch_size = datamodule.batch_size
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum

        self.model_config = model
        self.optimizer_config = optim
        self.datamodule_config = datamodule
        self.transform = transform

    def configure_optimizers(self):
        return self.optimizer

    def setup(self, stage):
        self.loss = nn.CrossEntropyLoss()

        self.anchor_encoder = hydra.utils.instantiate(self.model_config)
        self.sample_encoder = deepcopy(self.anchor_encoder)
        for param in self.sample_encoder.parameters():
            param.requires_grad = False

        self.optimizer = hydra.utils.instantiate(
            self.optimizer_config, self.anchor_encoder.parameters()
        )

        self.register_buffer("queue", torch.randn(self.embedding_dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _update_sample_encoder(self):
        for param, tracked_param in zip(
            self.sample_encoder.parameters(), self.anchor_encoder.parameters()
        ):
            param.data = (
                self.momentum * param.data + (1 - self.momentum) * tracked_param.data
            )

    def _encode(self, anchor_ids, positive_ids):
        anchors = self.anchor_encoder(anchor_ids)
        anchors = F.normalize(anchors)

        with torch.no_grad():
            self._update_sample_encoder()
            positives = self.sample_encoder(positive_ids)
            positives = F.normalize(positives)

        return anchors, positives

    def forward(self, anchor_tokens, sample_tokens):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, update_queue=True):
        assert batch["anchor"]["token_ids"].size(0) == batch["positive"][
            "token_ids"
        ].size(0)
        anchors, positives = self._encode(batch["anchor"], batch["positive"])
        positive_logits = torch.einsum("ij,ij->i", (anchors, positives)).unsqueeze(-1)
        negative_logits = torch.einsum("ij,jk", (anchors, self.queue.clone().detach()))

        logits = torch.cat((positive_logits, negative_logits), dim=1) / self.temperature
        labels = torch.zeros(positive_logits.size(0), dtype=torch.long)

        if update_queue:
            self._dequeue_and_enqueue(positives)

        loss = self.loss(logits, labels)
        return {"loss": loss, "log": {"loss": loss}}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, samples):
        samples = (
            samples.clone().detach()
        )  # for multi-gpu replace with: concat_all_gather(samples)
        batch_size = samples.size(0)

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the samples at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = samples.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr = torch.tensor(ptr, dtype=torch.long)
