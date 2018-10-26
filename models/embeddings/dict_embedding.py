#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.onnx.operators
from pytext.config.module_config import PoolingType


class DictEmbedding(nn.Embedding):
    def __init__(
        self,
        embed_num: int,
        embed_dim: int,
        pooling_type: PoolingType,
        sparse: bool = False,
    ) -> None:
        super().__init__(embed_num, embed_dim, sparse=sparse)
        self.pooling_type = pooling_type
        self.weight.data.uniform_(0, 0.1)
        self.sparse = sparse

    def forward(
        self,
        feats: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = torch.onnx.operators.shape_as_tensor(feats)[0]
        new_len_shape = torch.cat((batch_size.view(1), torch.LongTensor([-1])))
        lengths = torch.onnx.operators.reshape_from_tensor_shape(lengths, new_len_shape)
        max_toks = torch.onnx.operators.shape_as_tensor(lengths)[1]
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
