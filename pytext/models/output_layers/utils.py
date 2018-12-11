#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
from caffe2.python import core


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
