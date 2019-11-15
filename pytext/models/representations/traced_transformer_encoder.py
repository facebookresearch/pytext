#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import torch
import torch.nn as nn
from fairseq.modules import (
    TransformerSentenceEncoder as TransformerSentenceEncoderModule,
)


# Wrapper for TransformerSentenceEncoder to enable tracing
class TraceableTransformerWrapper(nn.Module):
    def __init__(self, eager_encoder: TransformerSentenceEncoderModule) -> None:
        super().__init__()
        assert hasattr(eager_encoder, "traceable")
        assert eager_encoder.traceable
        self.eager_encoder = eager_encoder

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        positions: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.eager_encoder(tokens, segment_labels, positions=positions)


class TracedTransformerEncoder(nn.Module):
    def __init__(
        self,
        eager_encoder: TransformerSentenceEncoderModule,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        positions: torch.Tensor = None,
    ):
        super().__init__()
        traceable_encoder = TraceableTransformerWrapper(eager_encoder)
        traced_encoder_inputs = self._prepare_inputs(tokens, segment_labels, positions)
        self.has_segment_labels = segment_labels is not None
        self.has_positions = positions is not None

        # do not check trace because of non-deterministic ops (e.g. dropout)
        self.traced_encoder = torch.jit.trace(
            traceable_encoder, tuple(traced_encoder_inputs), check_trace=False
        )

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        positions: torch.Tensor = None,
    ):
        assert self.has_segment_labels == (segment_labels is not None)
        assert self.has_positions == (positions is not None)

        traced_encoder_inputs = self._prepare_inputs(tokens, segment_labels, positions)
        encoded_layers, pooled_output = self.traced_encoder(*traced_encoder_inputs)
        encoded_layers = list(torch.unbind(encoded_layers))
        return encoded_layers, pooled_output

    def _prepare_inputs(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        positions: torch.Tensor = None,
    ):
        inputs = [tokens]
        if segment_labels is not None:
            inputs += [segment_labels]
        if positions is not None:
            inputs += [positions]
        return inputs
