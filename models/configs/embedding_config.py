#!/usr/bin/env python3

# TODO: convert these to ConfigBase types


class WordEmbeddingConfig:
    __slots__ = [
        "vocab_size",
        "embedding_dim",
        "pretrained_embeds_weight",
        "unk_idx",
        "sparse",
        "freeze_embeds",
        "embed_init_range",
    ]

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_embeds_weight,
        unk_idx,
        sparse,
        freeze_embeds,
        embed_init_range,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pretrained_embeds_weight = pretrained_embeds_weight
        self.unk_idx = unk_idx
        self.sparse = sparse
        self.freeze_embeds = freeze_embeds
        self.embed_init_range = embed_init_range


class DictEmbeddingConfig:
    __slots__ = ["vocab_size", "embedding_dim", "pooling_type", "sparse"]

    def __init__(self, vocab_size=0, embedding_dim=0, pooling_type=None, sparse=False):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type
        self.sparse = sparse


class CharacterEmbeddingConfig:
    __slots__ = [
        "vocab_size",
        "embedding_dim",
        "out_channels",
        "kernel_sizes",
        "sparse",
    ]

    def __init__(self, vocab_size, embedding_dim, out_channels, kernel_sizes, sparse):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.sparse = sparse
