#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

import torch
from torch import Tensor


class NAREncoderUtility:
    def prepare_for_nar_inference(
        self, length_beam_size: int, encoder_out: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        During masked NAR inference, multiple lengths are predicted for each
        item in the batch. Hence tiling has to be done in such a way that all new
        rows related to each item should be placed together. This is the assumption
        that we are following in the rest of the nar generation code.
        Eg:
        [row1, row2, row3] should be tiled as
        [row1, row1, row1, row2, row2, row2, row3, row3, row3]
        NOT [row1, row2, row3, row1, row2, row3, row1, row2, row3]
        """
        tiled_out = torch.jit.annotate(Dict[str, Tensor], {})
        bsz = encoder_out["encoder_out"].size(1)
        x = encoder_out["encoder_out"]
        new_x = (
            x.unsqueeze(2)
            .repeat(1, 1, length_beam_size, 1)
            .view(-1, bsz * length_beam_size, x.size(-1))
        )
        tiled_out["encoder_out"] = new_x

        if "encoder_mask" in encoder_out:
            new_encoder_mask = (
                encoder_out["encoder_mask"]
                .unsqueeze(1)
                .repeat(1, length_beam_size, 1)
                .view(bsz * length_beam_size, -1)
            )
            tiled_out["encoder_mask"] = new_encoder_mask

        if "src_tokens" in encoder_out:
            new_src_tokens = (
                encoder_out["src_tokens"]
                .unsqueeze(1)
                .repeat(1, length_beam_size, 1)
                .view(bsz * length_beam_size, -1)
            )
            tiled_out["src_tokens"] = new_src_tokens

        if "src_subword_begin_indices" in encoder_out:
            new_src_subword_begin_indices = (
                encoder_out["src_subword_begin_indices"]
                .unsqueeze(1)
                .repeat(1, length_beam_size, 1)
                .view(bsz * length_beam_size, -1)
            )
            tiled_out["src_subword_begin_indices"] = new_src_subword_begin_indices

        if "src_lengths" in encoder_out:
            new_src_lengths = (
                encoder_out["src_lengths"]
                .reshape(bsz, 1)
                .unsqueeze(0)
                .repeat(1, length_beam_size, 1)
                .view(bsz * length_beam_size, -1)
            )
            tiled_out["src_lengths"] = new_src_lengths

        if "src_index_tokens" in encoder_out:
            new_src_tokens = (
                encoder_out["src_index_tokens"]
                .unsqueeze(1)
                .repeat(1, length_beam_size, 1)
                .view(bsz * length_beam_size, -1)
            )
            tiled_out["src_index_tokens"] = new_src_tokens

        return tiled_out
