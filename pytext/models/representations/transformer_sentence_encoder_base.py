#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.models.representations.representation_base import RepresentationBase


class PoolingMethod(Enum):
    """
    Pooling Methods are chosen from the "Feature-based Approachs" section in
    https://arxiv.org/pdf/1810.04805.pdf
    """

    AVG_CONCAT_LAST_4_LAYERS = "avg_concat_last_4_layers"
    # since we only use hidden state of the first token from last layer for fine-tuning
    AVG_SECOND_TO_LAST_LAYER = "avg_second_to_last_layer"
    AVG_LAST_LAYER = "avg_last_layer"
    AVG_SUM_LAST_4_LAYERS = "avg_sum_last_4_layers"
    CLS_TOKEN = "cls_token"
    NO_POOL = "no_pool"


class TransformerSentenceEncoderBase(RepresentationBase):
    """
    Base class for all Bi-directional Transformer based Sentence Encoders. All
    children of this class should implement an _encoder function which takes
    as input: tokens, [optional] segment labels and a pad mask and outputs both
    the sentence representation (output of _pool_encoded_layers) and the
    output states of all the intermediate Transformer layers as a list of
    tensors.

    Input tuple consists of the following elements:
    1) tokens: torch tensor of size B x T which contains tokens ids
    2) pad_mask: torch tensor of size B x T generated with the condition
    tokens != self.vocab.get_pad_index()
    3) segment_labels: torch tensor of size B x T which contains the segment
    id of each token

    Output tuple consists of the following elements:
    1) encoded_layers: List of torch tensors where each tensor has shape
    B x T x C and there are num_transformer_layers + 1 of these.
    Each tensor represents the output of the intermediate
    transformer layers with the 0th element being the input to the
    first transformer layer (token + segment + position emebdding).
    2) [Optional] pooled_output: Output of the pooling operation associated
    with config.pooling_method to the encoded_layers.
    Size B x C (or B x 4C if pooling = AVG_CONCAT_LAST_4_LAYERS)
    """

    __EXPANSIBLE__ = True

    class Config(RepresentationBase.Config, ConfigBase):
        output_dropout: float = 0.4
        embedding_dim: int = 768
        pooling: PoolingMethod = PoolingMethod.CLS_TOKEN
        export: bool = False

    @classmethod
    def from_config(cls, config: Config, output_encoded_layers=False, *args, **kwargs):
        return cls(config, output_encoded_layers, *args, **kwargs)

    def __init__(
        self, config: Config, output_encoded_layers=False, *args, **kwargs
    ) -> None:
        super().__init__(config)
        self.pooling = config.pooling
        self.output_dropout = nn.Dropout(config.output_dropout)
        self.output_encoded_layers = output_encoded_layers
        self.export = config.export

        assert (
            self.pooling != PoolingMethod.NO_POOL or self.output_encoded_layers
        ), "If PoolingMethod is no_pool then output_encoded_layers should be True"

        if self.pooling == PoolingMethod.AVG_CONCAT_LAST_4_LAYERS:
            self.representation_dim = config.embedding_dim * 4
        else:
            self.representation_dim = config.embedding_dim

    def _encoder(
        self, input_tuple: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "Transformer Sentence Encoders should implement an encoder function"
        )

    def _pool_encoded_layers(
        self, encoded_layers: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == PoolingMethod.NO_POOL:
            return None
        elif self.pooling == PoolingMethod.AVG_CONCAT_LAST_4_LAYERS:
            sent_out = torch.cat(encoded_layers[-4:], 2)
        elif self.pooling == PoolingMethod.AVG_SUM_LAST_4_LAYERS:
            sent_out = torch.stack(encoded_layers[-4:]).sum(0)
        elif self.pooling == PoolingMethod.AVG_SECOND_TO_LAST_LAYER:
            sent_out = encoded_layers[-2]
        elif self.pooling == PoolingMethod.AVG_LAST_LAYER:
            sent_out = encoded_layers[-1]
        else:
            raise Exception("{} pooling is not supported".format(self.pooling))

        pad_mask = pad_mask.unsqueeze(2)
        sent_out = sent_out * pad_mask.float()
        pooled_output = torch.sum(sent_out, 1) / torch.sum(pad_mask, 1).float()
        return pooled_output

    def forward(
        self, input_tuple: Tuple[torch.Tensor, ...], *args
    ) -> Tuple[torch.Tensor, ...]:

        encoded_layers, pooled_output = self._encoder(input_tuple)

        pad_mask = input_tuple[1]

        if self.pooling != PoolingMethod.CLS_TOKEN:
            pooled_output = self._pool_encoded_layers(encoded_layers, pad_mask)

        if pooled_output is not None:
            pooled_output = self.output_dropout(pooled_output)

        output = []
        if self.output_encoded_layers:
            output.append(encoded_layers)
        if self.pooling != PoolingMethod.NO_POOL:
            output.append(pooled_output)
        return tuple(output)
