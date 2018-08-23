#!/usr/bin/env python3

#TODO: convert these to ConfigBase types


class WordEmbeddingConfig:
    __slots__ = [
        "vocab_size",
        "embedding_dim",
        "pretrained_embeds_weight",
        "unk_idx",
        "sparse_grad",
        "freeze_embeds",
    ]

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_embeds_weight,
        unk_idx,
        sparse_grad,
        freeze_embeds,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pretrained_embeds_weight = pretrained_embeds_weight
        self.unk_idx = unk_idx
        self.sparse_grad = sparse_grad
        self.freeze_embeds = freeze_embeds


class DictEmbeddingConfig:
    __slots__ = ["vocab_size", "embedding_dim", "pooling_type"]

    def __init__(self, vocab_size=0, embedding_dim=0, pooling_type=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type


class CharacterEmbeddingConfig:
    __slots__ = ["vocab_size", "embedding_dim", "out_channels", "kernel_sizes"]

    def __init__(self, vocab_size, embedding_dim, out_channels, kernel_sizes):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
