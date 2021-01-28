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
def validate_padding_control(padding_control: Optional[List[int]]) -> bool:
    if padding_control is not None:
        if len(padding_control) < 2:
            return False
        elif padding_control[0] != 0:
            return False

    return True


@torch.jit.script
def pad_length(
    len: int, padding_control: Optional[List[int]], max_len: int = -1
) -> int:
    if not validate_padding_control(padding_control):
        raise NotImplementedError

    if padding_control is not None:
        for pad in padding_control:
            if pad >= len:
                len = pad
                break
    if max_len > 0:
        len = min(len, max_len)
    return len


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
    input: List[List[int]],
    pad_value: int = 0,
    seq_padding_control: Optional[List[int]] = None,
    max_seq_pad_len: int = -1,
    batch_padding_control: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list to a 2d tensor. Returns a pair of tensors, the padded tensor
    as well as a mask tensor. The mask tensor has the same shape as the padded tensor,
    with a 1 in the position of non-pad values and a 0 in the position of pads.
    If padding_control is set, perform padding according to the specified padding style"""

    # List comprehension required for TorchScript
    max_seq_len = max([len(i) for i in input])  # noqa
    max_seq_len = pad_length(max_seq_len, seq_padding_control, max_seq_pad_len)

    max_batch_len = len(input)
    max_batch_len = pad_length(max_batch_len, batch_padding_control, -1)

    tensor = long_tensor_2d((max_batch_len, max_seq_len), pad_value)
    for i in range(len(input)):
        for j in range(min(len(input[i]), max_seq_len)):
            tensor[i][j] = input[i][j]
    mask = tensor.ne(pad_value).to(torch.long)
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
    batch: List[List[int]], seq_lens: List[int], pad_idx: int, max_len: int = -1
) -> List[List[int]]:
    pad_to_length = list_max(seq_lens)
    if max_len > 0:
        pad_to_length = min(pad_to_length, max_len)
    for sentence in batch:
        padding = pad_to_length - len(sentence)
        if padding >= 0:
            for _ in range(padding):
                sentence.append(pad_idx)
        else:
            for _ in range(-padding):
                sentence.pop()
    return batch


@torch.jit.script
def pad_3d(
    batch: List[List[List[int]]],
    tokens_lengths: List[List[int]],
    pad_idx: int,
) -> Tuple[List[List[List[int]]], List[List[int]]]:
    pad_to_1d: int = 0
    pad_to_2d: int = 0
    for tokens_length in tokens_lengths:
        pad_to_1d = max(pad_to_1d, len(tokens_length))
        pad_to_2d = max(pad_to_2d, list_max(tokens_length))
    for sentence, sentence_len in zip(batch, tokens_lengths):
        for _ in range(pad_to_1d - len(sentence)):
            new_list: List[int] = []
            sentence.append(new_list)
            sentence_len.append(0)
        for token in sentence:
            for _ in range(pad_to_2d - len(token)):
                token.append(pad_idx)
    return batch, tokens_lengths


@torch.jit.script
def pad_2d_float(
    batch: List[List[float]],
    seq_lens: List[int],
    pad_val: float = 0.0,
    max_len: int = -1,
) -> List[List[float]]:
    pad_to_length = list_max(seq_lens)
    if max_len > 0:
        pad_to_length = min(pad_to_length, max_len)
    for sentence in batch:
        padding = pad_to_length - len(sentence)
        if padding >= 0:
            for _ in range(padding):
                sentence.append(pad_val)
        else:
            for _ in range(-padding):
                sentence.pop()
    return batch


@torch.jit.script
def pad_3d_float(
    batch: List[List[List[float]]], seq_lens: List[int], pad_val: float = 0.0
) -> List[List[List[float]]]:
    outer_pad_to_length = list_max(seq_lens)
    inner_pad_to_length = -1
    for outer_list in batch:
        for inner_list in outer_list:
            inner_pad_to_length = max(inner_pad_to_length, len(inner_list))

    for outer_list in batch:
        for inner_list in outer_list:
            for _ in range(inner_pad_to_length - len(inner_list)):
                inner_list.append(pad_val)

        for _ in range(outer_pad_to_length - len(outer_list)):
            outer_list.append([pad_val] * inner_pad_to_length)

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
