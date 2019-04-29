#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.field_config import CharFeatConfig
from pytext.data.utils import Vocabulary
from pytext.fields import FieldMeta

from .embedding_base import EmbeddingBase


class CharacterEmbedding(EmbeddingBase):
    """
    Module for character aware CNN embeddings for tokens. It uses convolution
    followed by max-pooling over character embeddings to obtain an embedding
    vector for each token.

    Implementation is loosely based on https://arxiv.org/abs/1508.06615.

    Args:
        num_embeddings (int): Total number of characters (vocabulary size).
        embed_dim (int): Size of character embeddings to be passed to convolutions.
        out_channels (int): Number of output channels.
        kernel_sizes (List[int]): Dimension of input Tensor passed to MLP.
        highway_layers (int): Number of highway layers applied to pooled output.
        projection_dim (int): If specified, size of output embedding for token, via
            a linear projection from convolution output.

    Attributes:
        char_embed (nn.Embedding): Character embedding table.
        convs (nn.ModuleList): Convolution layers that operate on character
        embeddings.
        highway_layers (nn.Module): Highway layers on top of convolution output.
        projection (nn.Module): Final linear layer to token embedding.
        embedding_dim (int): Dimension of the final token embedding produced.

    """

    Config = CharFeatConfig

    @classmethod
    def from_config(
        cls,
        config: CharFeatConfig,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
    ):
        """Factory method to construct an instance of CharacterEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (CharFeatConfig): Configuration object specifying all the
                parameters of CharacterEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of CharacterEmbedding.

        """
        vocab_size = len(labels) if labels is not None else metadata.vocab_size
        return cls(
            vocab_size,
            config.embed_dim,
            config.cnn.kernel_num,
            config.cnn.kernel_sizes,
            config.highway_layers,
            config.projection_dim,
        )

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        out_channels: int,
        kernel_sizes: List[int],
        highway_layers: int,
        projection_dim: Optional[int],
        *args,
        **kwargs,
    ) -> None:
        conv_out_dim = len(kernel_sizes) * out_channels
        output_dim = projection_dim or conv_out_dim
        super().__init__(output_dim)

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
        self.highway = None
        if highway_layers > 0:
            self.highway = Highway(conv_out_dim, highway_layers)
        self.projection = None
        if projection_dim:
            self.projection = nn.Linear(conv_out_dim, projection_dim)

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

        # char_embedding: (bsize * max_sent_length, max_word_length, embed_dim)
        char_embedding = self.char_embed(chars)

        # conv_inp dim: (bsize * max_sent_length, emb_size, max_word_length)
        conv_inp = char_embedding.transpose(1, 2)
        char_conv_outs = [F.relu(conv(conv_inp)) for conv in self.convs]

        # Apply max pooling
        # char_pool_out[i] dims: (bsize * max_sent_length, out_channels)
        char_pool_outs = [torch.max(out, dim=2)[0] for out in char_conv_outs]

        # Concat different feature maps together
        # char_pool_out dim: (bsize * max_sent_length, out_channel * num_kernels)
        char_out = torch.cat(char_pool_outs, 1)

        # Highway layers, preserves dims
        if self.highway is not None:
            char_out = self.highway(char_out)

        if self.projection is not None:
            # Linear map back to final embedding size:
            # (bsize * max_sent_length, projection_dim)
            char_out = self.projection(char_out)

        # Reshape to (bsize, max_sent_length, "output_dim")
        return char_out.view(batch_size, max_sent_length, -1)


class Highway(nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`.
    Adopted from the AllenNLP implementation.
    """

    def __init__(self, input_dim: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            # As per comment in AllenNLP:
            # We should bias the highway layer to just carry its input forward. We do
            # that by setting the bias on `B(x)` to be positive, because that means `g`
            # will be biased to be high, so we will carry the input forward. The bias
            # on `B(x)` is the second half of the bias vector in each Linear layer.
            nn.init.constant_(layer.bias[self.input_dim :], 1)
            nn.init.constant_(layer.bias[: self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = F.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
        return x
