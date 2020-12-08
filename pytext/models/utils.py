#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch


def normalize_embeddings(embeddings: torch.Tensor):
    # assume [batch, embed_dim] dimensions
    # eps to make sure everything works in fp16
    return torch.nn.functional.normalize(embeddings, eps=1e-6)
