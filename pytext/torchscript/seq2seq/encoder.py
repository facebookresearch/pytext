#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Tuple

import torch
import torch.jit
import torch.jit.quantized
from torch import Tensor, nn


class EncoderEnsemble(nn.Module):
    """
    This class will call the encoders from all the models in the ensemble.
    It will process the encoder output to prepare input for each decoder step
    input
    """

    def __init__(self, models, beam_size):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.beam_size = beam_size

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        dict_feat: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        contextual_token_embedding: Optional[Tensor] = None,
    ) -> List[Dict[str, Tensor]]:
        # (seq_length, batch_size) for compatibility with Caffe2
        src_tokens_seq_first = src_tokens.t()

        # Model level parallelism cannot be implemented because of lack of
        # support for future type. https://github.com/pytorch/pytorch/issues/26578
        futures = torch.jit.annotate(List[Dict[str, Tensor]], [])

        for model in self.models:
            embedding_input = [[src_tokens_seq_first]]
            if dict_feat is not None:
                embedding_input.append(list(dict_feat))
            if contextual_token_embedding is not None:
                embedding_input.append([contextual_token_embedding])
            embeddings = model.source_embeddings(embedding_input)
            futures.append(
                # torch.jit._fork(model.encoder, src_tokens_seq_first, src_lengths)
                model.encoder(src_tokens_seq_first, embeddings, src_lengths)
            )
        return self.prepare_decoderstep_ip(futures)

    def prepare_decoderstep_ip(
        self, futures: List[Dict[str, Tensor]]
    ) -> List[Dict[str, Tensor]]:

        outputs = torch.jit.annotate(List[Dict[str, Tensor]], [])
        for idx, model in enumerate(self.models):
            # encoder_out = torch.jit._wait(future)
            encoder_out = futures[idx]
            tiled_encoder_out = model.encoder.tile_encoder_out(
                self.beam_size, encoder_out
            )
            outputs.append(tiled_encoder_out)
        return outputs
