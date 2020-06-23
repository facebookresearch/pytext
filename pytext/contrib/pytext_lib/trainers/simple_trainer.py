#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from statistics import mean

from pytext.utils import set_random_seeds


class SimpleTrainer:
    def __init__(self):
        # make result reproducible for testing purpose
        set_random_seeds(seed=0, use_deterministic_cudnn=True)

    def fit(self, dataloader, model, optimizer, epoch: int = 1):
        for i_epoch in range(epoch):
            model.train()
            losses = []
            for batch_id, batch in enumerate(dataloader):
                print(f"batch_id: {batch_id}", end="\r")
                optimizer.zero_grad()
                logits = model(batch)
                targets = batch["label_ids"]
                loss = model.get_loss(logits, targets)
                batch_size = len(targets)
                loss = loss / batch_size
                optimizer.backward(loss)
                optimizer.step()
                losses.append(loss.item())
            print("Epoch: {}, Train Loss: {}".format(i_epoch, mean(losses)))
