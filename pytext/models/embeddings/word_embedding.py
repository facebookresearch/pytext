#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
from typing import List, Optional

import torch
from pytext.config.field_config import WordFeatConfig
from pytext.data.tensorizers import Tensorizer
from pytext.data.utils import UNK
from pytext.fields import FieldMeta
from pytext.utils.embeddings import PretrainedEmbedding
from torch import nn

from .embedding_base import EmbeddingBase


class WordEmbedding(EmbeddingBase):
    """
    A word embedding wrapper module around `torch.nn.Embedding` with options to
    initialize the word embedding weights and add MLP layers acting on each word.

    Note: Embedding weights for UNK token are always initialized to zeros.

    Args:
        num_embeddings (int): Total number of words/tokens (vocabulary size).
        embedding_dim (int): Size of embedding vector.
        embeddings_weight (torch.Tensor): Pretrained weights to initialize the
            embedding table with.
        init_range (List[int]): Range of uniform distribution to initialize the
            weights with if `embeddings_weight` is None.
        unk_token_idx (int): Index of UNK token in the word vocabulary.
        mlp_layer_dims (List[int]): List of layer dimensions (if any) to add
            on top of the embedding lookup.

    """

    Config = WordFeatConfig

    @classmethod
    def from_config(
        cls,
        config: WordFeatConfig,
        metadata: Optional[FieldMeta] = None,
        tensorizer: Optional[Tensorizer] = None,
        init_from_saved_state: Optional[bool] = False,
    ):
        """Factory method to construct an instance of WordEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (WordFeatConfig): Configuration object specifying all the
            parameters of WordEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of WordEmbedding.

        """
        if tensorizer is not None:
            embeddings_weight = None
            if config.pretrained_embeddings_path and (
                # We don't need to load pretrained embeddings if we know the
                # embedding weights are going to be loaded from a snapshot. The
                # exception is if we rely on the pretrained embeddings to give us
                # the vocab, in which case, we have to load it regardless.
                config.vocab_from_pretrained_embeddings
                or not init_from_saved_state
            ):
                pretrained_embedding = PretrainedEmbedding(
                    config.pretrained_embeddings_path,  # doesn't support fbpkg
                    lowercase_tokens=config.lowercase_tokens,
                )

                if config.vocab_from_pretrained_embeddings:
                    # pretrained embeddings will get a freq count of 1
                    assert config.min_freq == 1, (
                        "If `vocab_from_pretrained_embeddings` is set, the vocab's "
                        "`min_freq` must be 1"
                    )
                    if not config.vocab_from_train_data:  # Reset token counter.
                        tensorizer.vocab_builder._counter = collections.Counter()
                    pretrained_vocab = pretrained_embedding.embed_vocab
                    if config.vocab_size:
                        pretrained_vocab = pretrained_vocab[: config.vocab_size]
                    tensorizer.vocab_builder.add_all(pretrained_vocab)
                    tensorizer.vocab = tensorizer.vocab_builder.make_vocab()

                embeddings_weight = pretrained_embedding.initialize_embeddings_weights(
                    tensorizer.vocab.idx,
                    UNK,
                    config.embed_dim,
                    config.embedding_init_strategy,
                )
            num_embeddings = len(tensorizer.vocab)
            unk_token_idx = tensorizer.vocab.idx[UNK]
        else:  # This else condition should go away after metadata goes away.
            num_embeddings = metadata.vocab_size
            embeddings_weight = metadata.pretrained_embeds_weight
            unk_token_idx = metadata.unk_token_idx

        return cls(
            num_embeddings=num_embeddings,
            embedding_dim=config.embed_dim,
            embeddings_weight=embeddings_weight,
            init_range=config.embedding_init_range,
            unk_token_idx=unk_token_idx,
            mlp_layer_dims=config.mlp_layer_dims,
        )

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 300,
        embeddings_weight: Optional[torch.Tensor] = None,
        init_range: Optional[List[int]] = None,
        unk_token_idx: int = 0,
        mlp_layer_dims: List[int] = (),
    ) -> None:
        output_embedding_dim = mlp_layer_dims[-1] if mlp_layer_dims else embedding_dim
        EmbeddingBase.__init__(self, embedding_dim=output_embedding_dim)

        # Create word embedding
        self.word_embedding = nn.Embedding(
            num_embeddings, embedding_dim, _weight=embeddings_weight
        )
        if embeddings_weight is None and init_range:
            self.word_embedding.weight.data.uniform_(init_range[0], init_range[1])
        # Initialize unk embedding with zeros
        # to guard the model against randomized decisions based on unknown words
        self.word_embedding.weight.data[unk_token_idx].fill_(0.0)

        # Create MLP layers
        if mlp_layer_dims is None:
            mlp_layer_dims = []
        self.mlp = nn.Sequential(
            *(
                nn.Sequential(nn.Linear(m, n), nn.ReLU())
                for m, n in zip([embedding_dim] + list(mlp_layer_dims), mlp_layer_dims)
            )
        )

    def __getattr__(self, name):
        if name == "weight":
            return self.word_embedding.weight
        return super().__getattr__(name)

    def forward(self, input):
        return self.mlp(self.word_embedding(input))

    def freeze(self):
        for param in self.word_embedding.parameters():
            param.requires_grad = False
