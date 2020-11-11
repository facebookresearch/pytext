#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Tuple
from pytext.torchscript.tensorizer.tensorizer import ScriptTensorizer

import torch


def max_tokens(per_sentence_tokens: List[List[Tuple[str, int, int]]]) -> int:
    """receive the tokenize output for a batch per_sentence_tokens,
    return the max token length of any sentence"""

    if len(per_sentence_tokens) == 0:
        return 0

    sentence_lengths = [len(sentence) for sentence in per_sentence_tokens]
    return max(sentence_lengths)


########################################################################
#
# utility functions to destructure flat result tensor combining
#   cross-request batches and client side
#   batches into a cross-request list of
#   client-side batch tensors
#


def destructure_tensor(
    client_batch: List[int],
    result_tensor: torch.Tensor,
) -> List[torch.Tensor]:
    start = 0
    res_list: List[torch.Tensor] = []

    for elems in client_batch:
        end = start + elems
        res_list.append(result_tensor.narrow(0, start, elems))
        start = end

    return res_list


def destructure_tensor_list(
    client_batch: List[int],
    result_tensor_list: List[torch.Tensor],
) -> List[List[torch.Tensor]]:
    res_list: List[List[torch.Tensor]] = []
    start = 0

    for elems in client_batch:
        end = start + elems
        res_list.append(result_tensor_list[start:end])
        start = end

    return res_list


############################################################################
#
# make_prediction_* ()
# utility functions to collect inputs from multiple batch elements into a
# a single cross request batch
#
# make_batch_* ()
# utility functions for batch optimizations
#


def make_prediction_texts(
    batch: List[
        Tuple[
            List[str],  # texts
        ]
    ],
) -> List[str]:

    batchsize = len(batch)
    flat_texts: List[str] = []

    for i in range(batchsize):
        batch_element = batch[i][0]
        flat_texts.extend(batch_element)

    if len(flat_texts) == 0:
        raise RuntimeError("This is not good. Empty request batch.")

    return flat_texts


def make_batch_texts(
    tensorizer: ScriptTensorizer,
    mega_batch: List[
        Tuple[
            List[str],  # texts
            int,
        ]
    ],
    goals: Dict[str, str],
) -> List[List[Tuple[List[str], int,]]]:  # texts

    # The next lines sort all cross-request batch elements by the token length.
    # Note that cross-request batch element can in turn be a client batch.
    mega_batch_key_list = [
        (max_tokens(tensorizer.tokenize(x[0], None)), n)
        for (n, x) in enumerate(mega_batch)
    ]
    sorted_mega_batch_key_list = sorted(mega_batch_key_list)
    sorted_mega_batch = [mega_batch[n] for (_, n) in sorted_mega_batch_key_list]

    # TBD: allow model server to specify batch size in goals dictionary
    max_bs: int = 10
    len_mb = len(mega_batch)
    num_batches = (len_mb + max_bs - 1) // max_bs

    batch_list: List[
        List[
            Tuple[
                List[str],  # texts
                int,  # position
            ]
        ]
    ] = []

    start = 0

    for _i in range(num_batches):
        end = min(start + max_bs, len_mb)
        batch_list.append(sorted_mega_batch[start:end])
        start = end

    return batch_list


#


def make_prediction_texts_wdense(
    batch: List[
        Tuple[
            List[str],  # texts
            List[List[float]],  # dense
        ]
    ],
) -> List[str]:

    batchsize = len(batch)
    flat_texts: List[str] = []

    for i in range(batchsize):
        batch_element = batch[i][0]
        flat_texts.extend(batch_element)

        if len(batch[i][0]) != len(batch[i][1]):
            raise RuntimeError(
                "This is not good. texts/dense client batch length mismatch"
            )

    if len(flat_texts) == 0:
        raise RuntimeError("This is not good. Empty request batch.")

    return flat_texts


def make_prediction_wtexts_dense(
    batch: List[
        Tuple[
            List[str],  # texts
            List[List[float]],  # dense
        ]
    ],
) -> List[List[float]]:

    batchsize = len(batch)
    flat_dense: List[List[float]] = []

    for i in range(batchsize):
        batch_element = batch[i][1]
        flat_dense.extend(batch_element)

        if len(batch[i][0]) != len(batch[i][1]):
            raise RuntimeError(
                "This is not good. texts/dense client batch length mismatch"
            )

    if len(flat_dense) == 0:
        raise RuntimeError("This is not good. Empty request batch.")

    return flat_dense


def make_batch_texts_dense(
    tensorizer: ScriptTensorizer,
    mega_batch: List[
        Tuple[
            List[str],  # texts
            List[List[float]],  # dense
            int,
        ]
    ],
    goals: Dict[str, str],
) -> List[List[Tuple[List[str], List[List[float]], int]]]:  # texts, dense, ??

    # The next lines sort all cross-request batch elements by the token length.
    # Note that cross-request batch element can in turn be a client batch.
    mega_batch_key_list = [
        (max_tokens(tensorizer.tokenize(x[0], None)), n)
        for (n, x) in enumerate(mega_batch)
    ]
    sorted_mega_batch_key_list = sorted(mega_batch_key_list)
    sorted_mega_batch = [mega_batch[n] for (_, n) in sorted_mega_batch_key_list]

    # TBD: allow model server to specify batch size in goals dictionary
    max_bs: int = 10
    len_mb = len(mega_batch)
    num_batches = (len_mb + max_bs - 1) // max_bs

    batch_list: List[
        List[
            Tuple[
                List[str],  # texts
                int,  # position
            ]
        ]
    ] = []

    start = 0

    for _i in range(num_batches):
        end = min(start + max_bs, len_mb)
        batch_list.append(sorted_mega_batch[start:end])
        start = end

    return batch_list


#
