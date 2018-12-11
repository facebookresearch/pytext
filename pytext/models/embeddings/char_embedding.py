#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.field_config import CharFeatConfig
from pytext.fields import FieldMeta

from .embedding_base import EmbeddingBase


class CharacterEmbedding(EmbeddingBase):
    """
    Module for character aware CNN embeddings for tokens. It uses convolution
    followed by max-pooling over character embeddings to obtain an embedding
    vector for each token.

    Implementation is loosely based on https://arxiv.org/abs/1508.06615 but,
    does not implement the Highway Network illustrated in the paper.

    Args:
        num_embeddings (int): Total number of characters (vocabulary size).
        embed_dim (int): Size of embedding vector.
        out_channels (int): Number of output channels.
        kernel_sizes (List[int]): Dimension of input Tensor passed to MLP.

    Attributes:
        char_embed (nn.Embedding): Character embedding table.
        convs (nn.ModuleList): Convolution layers that operate on character
        embeddings.
        embedding_dim (int): Dimension of the final token embedding produced.

    """

    Config = CharFeatConfig

    @classmethod
    def from_config(cls, config: CharFeatConfig, metadata: FieldMeta):
        """Factory method to construct an instance of CharacterEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (CharFeatConfig): Configuration object specifying all the
                parameters of CharacterEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of CharacterEmbedding.

        """
        return cls(
            metadata.vocab_size,
            config.embed_dim,
            config.cnn.kernel_num,
            config.cnn.kernel_sizes,
        )

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        out_channels: int,
        kernel_sizes: List[int],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(embed_dim)
        self.char_embed = nn.Embedding(num_embeddings, embed_dim)
        self.convs = nn.ModuleList(
            [
                # in_channels = embed_dim because input is treated as sequence
                # of dim [max_word_length] with embed_dim channels
                # Adding padding to provide robustness in cases where input
                # length is less than conv filter width
                nn.Conv1d(embed_dim, out_channels, K, padding=K // 2)
                for K in kernel_sizes
            ]
        )
        self.embedding_dim = out_channels * len(kernel_sizes)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of sentences such that tokens are broken into character ids,
        produce token embedding vectors for each sentence in the batch.

        Args:
            chars (torch.Tensor): Batch of sentences where each token is broken
            into characters.
            Dimension: batch size X maximum sentence length X maximum word length

        Returns:
            torch.Tensor: Embedded batch of sentences. Dimension:
            batch size X maximum sentence length, token embedding size.
            Token embedding size = `out_channels * len(self.convs))`

        """
        batch_size, max_sent_length, max_word_length = tuple(chars.size())
        chars = chars.view(batch_size * max_sent_length, max_word_length)

        # char_embedding: (bsize * max_sent_length, max_word_length, emb_size)
        char_embedding = self.char_embed(chars)

        # conv_inp dim: (bsize * max_sent_length, emb_size, max_word_length)
        conv_inp = char_embedding.transpose(1, 2)
        char_conv_outs = [F.relu(conv(conv_inp)) for conv in self.convs]

        # Apply max pooling
        # char_pool_out[i] dims: (bsize * max_sent_length, out_channels)
        char_pool_outs = [torch.max(out, dim=2)[0] for out in char_conv_outs]

        # Concat different feature maps together
        # char_pool_out dim: (bsize * max_sent_length, out_channel * num_kernels)
        char_pool_out = torch.cat(char_pool_outs, 1)

        # Reshape to (bsize, max_sent_length, out_channel * len(self.convs))
        return char_pool_out.view(batch_size, max_sent_length, -1)
