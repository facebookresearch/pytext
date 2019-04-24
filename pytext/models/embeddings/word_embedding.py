#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.field_config import WordFeatConfig
from pytext.data.tensorizers import Tensorizer
from pytext.data.utils import UNK, VocabBuilder
from pytext.fields import FieldMeta
from pytext.utils.embeddings import PretrainedEmbedding

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
        cls, config: WordFeatConfig, metadata: FieldMeta, tensorizer: Tensorizer = None
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
        # Backward compatiblity until we do away with metadata
        if metadata is not None:
            num_embeddings = metadata.vocab_size
            embeddings_weight = metadata.pretrained_embeds_weight
            unk_token_idx = metadata.unk_token_idx
        elif tensorizer is not None:
            embeddings_weight = None
            if config.pretrained_embeddings_path:
                pretrained_embedding = PretrainedEmbedding(
                    config.pretrained_embeddings_path,  # doesn't support fbpkg
                    lowercase_tokens=tensorizer.tokenizer.lowercase,
                )
                if config.vocab_from_pretrained_embeddings:
                    if not config.vocab_from_train_data:
                        tensorizer.vocab_builder = VocabBuilder()
                    tensorizer.vocab_builder.add_all(pretrained_embedding.embed_vocab)
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
            raise ValueError(
                "metadata and tensorizer objects both are None. "
                "WordEmbedding cannot be initalized."
            )

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
        embedding_dim: int,
        embeddings_weight: torch.Tensor,
        init_range: List[int],
        unk_token_idx: int,
        mlp_layer_dims: List[int],
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
        self.mlp_layers = nn.ModuleList([])
        for next_dim in mlp_layer_dims or []:
            assert next_dim > 0
            self.mlp_layers.append(nn.Linear(embedding_dim, next_dim))
            embedding_dim = next_dim

    def __getattr__(self, name):
        if name == "weight":
            return self.word_embedding.weight
        return super().__getattr__(name)

    def forward(self, input):
        embedding = self.word_embedding(input)

        for mlp_layer in self.mlp_layers:
            embedding = mlp_layer(embedding)
            embedding = F.relu(embedding)

        return embedding

    def freeze(self):
        for param in self.word_embedding.parameters():
            param.requires_grad = False
