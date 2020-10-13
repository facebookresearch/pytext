#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
from caffe2.python import core
from torch import Tensor


class OutputLayerUtils:
    @staticmethod
    def gen_additional_blobs(
        predict_net: core.Net,
        probability_out,
        model_out: torch.Tensor,
        output_name: str,
        label_names: List[str],
    ) -> List[core.BlobReference]:
        """
        Utility method to generate additional blobs for human readable result for
        models that use explicit labels.
        """
        res = []
        tmp_out_score = predict_net.Log(probability_out)
        label_scores = predict_net.Split(
            tmp_out_score, label_names, axis=model_out.dim() - 1
        )

        # Make sure label_scores is iterable
        if not isinstance(label_scores, tuple):
            label_scores = (label_scores,)
        for name, label_score in zip(label_names, label_scores):
            res.append(predict_net.Copy(label_score, "{}:{}".format(output_name, name)))
        return res


def query_word_reprs(encoder_repr: Tensor, token_indices: Tensor) -> Tensor:
    """
    Given an encoder_repr (B x T_1 x H) and token_indices (B x T_2) where T_2 <= T_1,
    collect embeddings from encoder_repr pertaining to indices in token_indices. In the
    context of fine-tuning pre-trained encoders on sequence labeling, our goal is to
    build token-level representations as opposed to subword-level represenatations
    for alignment with other token-level cues, such as dictionary features. Currently,
    a token representation is built by taking its first subword representation.
    """

    return torch.gather(
        encoder_repr,
        1,
        token_indices.unsqueeze(2).expand(-1, -1, encoder_repr.size(-1)),
    )
