#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterEmbedding(nn.Module):
    """Implements character aware CNN embeddings for tokens.

    Implementation is loosely based on https://arxiv.org/abs/1508.06615 and
    does not implement the Highway Network illustrated in the paper.
    """

    def __init__(
        self,
        embed_num: int,
        embed_dim: int,
        out_channels: int,
        kernel_sizes: List[int],
        sparse=False,
    ) -> None:
        super().__init__()
        self.char_embed = nn.Embedding(embed_num, embed_dim, sparse=sparse)
        self.convs = nn.ModuleList(
            [
                # in_channels = embed_dim because input is treated as sequence
                # of dim [max_word_length] with embed_dim channels
                nn.Conv1d(embed_dim, out_channels, K)
                for K in kernel_sizes
            ]
        )
        # TODO: Revisit this if ONNX can't handle it; use nn.max instead
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.embedding_dim = out_channels * len(kernel_sizes)
        self.sparse = sparse

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        # chars: (bsize, max_sent_length, max_word_length)
        batch_size, max_sent_length, max_word_length = tuple(chars.size())
        chars = chars.view(batch_size * max_sent_length, max_word_length)

        # char_embedding: (bsize * max_sent_length, max_word_length, emb_size)
        char_embedding = self.char_embed(chars)

        # conv_inp dim: (bsize * max_sent_length, emb_size, max_word_length)
        conv_inp = char_embedding.transpose(1, 2)
        char_conv_outs = [F.relu(conv(conv_inp)) for conv in self.convs]

        # Apply max pooling
        # char_pool_out[i] dims: (bsize * max_sent_length, out_channels)
        char_pool_outs = [self.pool(out).squeeze() for out in char_conv_outs]

        # Concat different feature maps together
        # char_pool_out dim: (bsize * max_sent_length, out_channel * num_kernels)
        char_pool_out = torch.cat(char_pool_outs, 1)

        # Reshape to (bsize, max_sent_length, out_channel * len(self.convs))
        return char_pool_out.view(batch_size, max_sent_length, -1)
