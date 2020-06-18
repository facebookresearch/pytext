#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor


class ScriptBatchInput(NamedTuple):
    """A batch of inputs for TorchScript Module(bundle of Tensorizer and Model)
    texts or tokens is required but multually exclusive
    Args:
        texts: a batch of raw text inputs
        tokens: a batch of pre-tokenized inputs
        languages: language for each input in the batch
    """

    texts: Optional[List[List[str]]]
    tokens: Optional[List[List[List[str]]]]
    languages: Optional[List[List[str]]]


# ===== the following section should be replaced once JIT provide native support
@torch.jit.script
def list_max(l: List[int]):
    max_value = l[0]  # fine to throw if empty
    for i in range(len(l) - 1):  # don't forget the +1
        max_value = max(max_value, l[i + 1])
    return max_value


@torch.jit.script
def list_str_index(l: List[str], element: str, start: int) -> int:
    """
    Equivalent to: list.index(v, start)
    """
    for i, t in enumerate(l[start:]):
        if t == element:
            return start + i
    return -1


@torch.jit.script
def list_membership(item: int, list: List[int]):
    item_present = False
    for i in list:
        if item == i:
            item_present = True
    return item_present


@torch.jit.script
def reverse_tensor_list(int_list: List[torch.Tensor]) -> List[torch.Tensor]:
    l_len = len(int_list)
    res = []
    for idx in range(l_len):
        res.append(int_list[l_len - idx - 1])
    return res


@torch.jit.script
def long_tensor_2d(shape: Tuple[int, int], fill_value: int = 0) -> torch.Tensor:
    """Return a new 2d torch.LongTensor with size according to shape.
    The values of this tensor will be fill_value."""
    outer = torch.jit.annotate(List[List[int]], [])
    inner = torch.jit.annotate(List[int], [])
    for _i in range(shape[1]):
        inner.append(fill_value)
    for _i in range(shape[0]):
        outer.append(inner)
    return torch.tensor(outer, dtype=torch.long)


@torch.jit.script
def pad_2d_mask(
    input: List[List[int]], pad_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list to a 2d tensor. Returns a pair of tensors, the padded tensor
    as well as a mask tensor. The mask tensor has the same shape as the padded tensor,
    with a 1 in the position of non-pad values and a 0 in the position of pads."""
    max_len = 0
    for i in input:
        max_len = max(max_len, len(i))
    tensor = long_tensor_2d((len(input), max_len), pad_value)
    mask = long_tensor_2d((len(input), max_len), 0)
    for i in range(len(input)):
        for j in range(len(input[i])):
            tensor[i][j] = input[i][j]
            mask[i][j] = 1
    return tensor, mask


# ========= end section


@torch.jit.script
def utf8_chars(s: str) -> List[str]:
    """An implementation of UTF8 character iteration in TorchScript.
    There are no bitwise operations in torchscript, so we compare directly to
    integer values. There isn't a lot of validation, for instance if you pass
    in an improperly encoded string with an out-of-place continuation byte,
    or with a non-left-to-right byte order, you'll get unexpected results
    and likely throw. Torch itself takes in unicode strings and encodes them
    as UTF8, so that should be actively hard to do.

    The logic is simple: looking at the current start-of-character byte.
    If its high bit is 0, it's a 1-byte character. Otherwise, the number of
    bytes is the number of leading 1s in its binary representation, so
    find that number by comparing it directly to ints with the appropriate
    representation, then append that many bytes as a character and move past
    them to the next start byte.
    """
    chars = torch.jit.annotate(List[str], [])
    i = 0
    while i < len(s):
        byte = ord(s[i])
        if byte < 0b10000000:
            chars.append(s[i])
            i += 1
        else:
            if byte < 0b11100000:
                num_bytes = 2
            elif byte < 0b11110000:
                num_bytes = 3
            elif byte < 0b11111000:
                num_bytes = 4
            elif byte < 0b11111100:
                num_bytes = 5
            elif byte < 0b11111110:
                num_bytes = 6
            elif byte < 0b11111111:
                num_bytes = 7
            else:
                num_bytes = 8
            chars.append(s[i : i + num_bytes])
            i += num_bytes
    return chars


@torch.jit.script
def truncate_tokens(
    batch: List[List[str]], max_seq_len: int, pad_token: str
) -> List[List[str]]:
    truncated: List[List[str]] = []
    for sentence in batch:
        if len(sentence) == 0:
            truncated.append([pad_token])
        elif max_seq_len > 0:
            truncated.append(sentence[0:max_seq_len])
        else:
            truncated.append(sentence)
    return truncated


@torch.jit.script
def make_sequence_lengths(batch: List[List[str]]) -> List[int]:
    seq_lens = torch.jit.annotate(List[int], [])
    for sentence in batch:
        seq_lens.append(len(sentence))
    return seq_lens


@torch.jit.script
def pad_2d(
    batch: List[List[int]], seq_lens: List[int], pad_idx: int
) -> List[List[int]]:
    pad_to_length = list_max(seq_lens)
    for sentence in batch:
        for _ in range(pad_to_length - len(sentence)):
            sentence.append(pad_idx)
    return batch


@torch.jit.script
def pad_2d_float(
    batch: List[List[float]], seq_lens: List[int], pad_val: float = 0.0
) -> List[List[float]]:
    pad_to_length = list_max(seq_lens)
    for sentence in batch:
        for _ in range(pad_to_length - len(sentence)):
            sentence.append(pad_val)
    return batch


@torch.jit.script
def add_special_token_2d(
    values: List[List[int]], special_token: int = 0, use_first_value: bool = False
) -> List[List[int]]:
    results = torch.jit.annotate(List[List[int]], [])
    for value in values:
        result = torch.jit.annotate(List[int], [])
        if use_first_value and len(value) > 0:
            special_token = value[0]
        result.append(special_token)
        result.extend(value)
        result.append(special_token)
        results.append(result)
    return results


@torch.jit.script
def add_bos_eos_2d(values: List[List[int]], bos: int, eos: int) -> List[List[int]]:
    results = torch.jit.annotate(List[List[int]], [])
    for value in values:
        result = torch.jit.annotate(List[int], [])
        result.append(bos)
        result.extend(value)
        result.append(eos)
        results.append(result)
    return results


@torch.jit.script
def make_byte_inputs(
    batch: List[List[str]], max_byte_len: int, offset_for_non_padding: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_lens = make_sequence_lengths(batch)
    max_num_tokens = list_max(seq_lens)
    bytes = torch.zeros(len(batch), max_num_tokens, max_byte_len, dtype=torch.long)

    for batch_index in range(len(batch)):
        sentence = batch[batch_index]
        for token_index in range(len(sentence)):
            token = sentence[token_index]
            for byte_index in range(min(len(token), max_byte_len)):
                s = token[byte_index]
                # use empty string as eos because torchscript not support chr()
                if s == "":
                    v = 256
                else:
                    v = ord(s)
                # add offset_for_non_padding to conform to Fairseq pre-training
                bytes[batch_index][token_index][byte_index] = v + offset_for_non_padding

    return bytes, torch.tensor(seq_lens)


@torch.jit.script
def squeeze_1d(inputs: Optional[List[str]]) -> Optional[List[List[str]]]:
    result: Optional[List[List[str]]] = None
    if inputs is not None:
        result = torch.jit.annotate(List[List[str]], [])
        for line in inputs:
            result.append([line])
    return result


@torch.jit.script
def squeeze_2d(inputs: Optional[List[List[str]]]) -> Optional[List[List[List[str]]]]:
    result: Optional[List[List[List[str]]]] = None
    if inputs is not None:
        result = torch.jit.annotate(List[List[List[str]]], [])
        for line in inputs:
            result.append([line])
    return result


def float_tensor_list1D(input_tensor: Tensor) -> List[float]:
    result: List[float] = []
    assert len(input_tensor.size()) == 1
    for idx in range(input_tensor.size(0)):
        result.append(float(input_tensor[idx]))
    return result
