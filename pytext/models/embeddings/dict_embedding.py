#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional

import torch
import torch.nn as nn
import torch.onnx.operators
from pytext.config.field_config import DictFeatConfig
from pytext.config.module_config import PoolingType
from pytext.data.tensorizers import Tensorizer
from pytext.data.utils import PAD_INDEX, UNK_INDEX, Vocabulary
from pytext.fields import FieldMeta
from pytext.utils import cuda

from .embedding_base import EmbeddingBase


class DictEmbedding(EmbeddingBase, nn.Embedding):
    """
    Module for dictionary feature embeddings for tokens. Dictionary features are
    also known as gazetteer features. These are per token discrete features that
    the module learns embeddings for.
    Example: For the utterance *Order coffee from Starbucks*, the dictionary
    features could be
    ::

        [
            {"tokenIdx": 1, "features": {"drink/beverage": 0.8, "music/song": 0.2}},
            {"tokenIdx": 3, "features": {"store/coffee_shop": 1.0}}
        ]

    ::
    Thus, for a given token there can be more than one dictionary features each
    of which has a confidence score. The final embedding for a token is the
    weighted average of the dictionary embeddings followed by a pooling operation
    such that the module produces an embedding vector per token.

    Args:
        num_embeddings (int): Total number of dictionary features (vocabulary size).
        embed_dim (int): Size of embedding vector.
        pooling_type (PoolingType): Type of pooling for combining the dictionary
            feature embeddings.

    Attributes:
        pooling_type (PoolingType): Type of pooling for combining the dictionary
            feature embeddings.

    """

    Config = DictFeatConfig

    @classmethod
    def from_config(
        cls,
        config: DictFeatConfig,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
        tensorizer: Optional[Tensorizer] = None,
    ):
        """Factory method to construct an instance of DictEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (DictFeatConfig): Configuration object specifying all the
            parameters of DictEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of DictEmbedding.

        """
        # TODO: clean this up once fully migrated to new data handler design
        vocab_size = (
            len(tensorizer.vocab)
            if tensorizer is not None
            else len(labels)
            if labels is not None
            else metadata.vocab_size
        )
        tensorizer_vocab_exists = tensorizer and tensorizer.vocab
        pad_index = (
            tensorizer.vocab.get_pad_index() if tensorizer_vocab_exists else PAD_INDEX
        )
        unk_index = (
            tensorizer.vocab.get_unk_index() if tensorizer_vocab_exists else UNK_INDEX
        )
        return cls(
            num_embeddings=vocab_size,
            embed_dim=config.embed_dim,
            pooling_type=config.pooling,
            pad_index=pad_index,
            unk_index=unk_index,
            mobile=config.mobile,
        )

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        pooling_type: PoolingType,
        pad_index: int = PAD_INDEX,
        unk_index: int = UNK_INDEX,
        mobile: bool = False,
    ) -> None:
        self.pad_index = pad_index
        self.unk_index = unk_index
        EmbeddingBase.__init__(self, embed_dim)
        nn.Embedding.__init__(
            self, num_embeddings, embed_dim, padding_idx=self.pad_index
        )
        self.pooling_type = pooling_type
        self.mobile = mobile

    def find_and_replace(
        self, tensor: torch.Tensor, find_val: int, replace_val: int
    ) -> torch.Tensor:
        """
        `torch.where` is not supported for mobile ONNX, this hack allows a mobile
        exported version of `torch.where` which is computationally more expensive
        """
        if self.mobile:
            mask = torch.eq(tensor, find_val)
            return tensor * (1 - mask.long()) + mask * replace_val
        else:
            return torch.where(
                tensor == find_val,
                cuda.GetTensor(torch.full_like(tensor, replace_val)),
                tensor,
            )

    def forward(
        self, feats: torch.Tensor, weights: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Given a batch of sentences such containing dictionary feature ids per
        token, produce token embedding vectors for each sentence in the batch.

        Args:
            feats (torch.Tensor): Batch of sentences with dictionary feature ids.
                shape: [bsz, seq_len * max_feat_per_token]
            weights (torch.Tensor): Batch of sentences with dictionary feature
                weights for the dictionary features.
                shape: [bsz, seq_len * max_feat_per_token]
            lengths (torch.Tensor): Batch of sentences with the number of
                dictionary features per token.
                shape: [bsz, seq_len]

        Returns:
            torch.Tensor: Embedded batch of sentences. Dimension:
            batch size X maximum sentence length, token embedding size.
            Token embedding size = `embed_dim` passed to the constructor.

        """
        batch_size = torch.onnx.operators.shape_as_tensor(feats)[0]
        max_toks = torch.onnx.operators.shape_as_tensor(lengths)[1]

        if self.unk_index != self.pad_index:
            # convert all unk indices to pad indices
            feats = self.find_and_replace(feats, self.unk_index, self.pad_index)

        dict_emb = super().forward(feats)

        # Calculate weighted average of the embeddings
        weighted_embds = dict_emb * weights.unsqueeze(2)
        new_emb_shape = torch.cat(
            (
                batch_size.view(1),
                max_toks.view(1),
                torch.LongTensor([-1]),
                torch.LongTensor([weighted_embds.size()[-1]]),
            )
        )
        weighted_embds = torch.onnx.operators.reshape_from_tensor_shape(
            weighted_embds, new_emb_shape
        )

        if self.pooling_type == PoolingType.MEAN:
            reduced_embeds = (
                torch.sum(weighted_embds, dim=2) / lengths.unsqueeze(2).float()
            )
        else:
            reduced_embeds, _ = torch.max(weighted_embds, dim=2)

        return reduced_embeds
