#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


CUDA_ENABLED = False
DISTRIBUTED_WORLD_SIZE = 1


def Variable(data, *args, **kwargs):
    if CUDA_ENABLED:
        return torch.autograd.Variable(data.cuda(), *args, **kwargs)
    else:
        return torch.autograd.Variable(data, *args, **kwargs)


def var_to_numpy(v):
    return (v.cpu() if CUDA_ENABLED else v).data.numpy()


def zerovar(*size):
    return Variable(torch.zeros(*size))


def FloatTensor(*args):
    if CUDA_ENABLED:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


def LongTensor(*args):
    if CUDA_ENABLED:
        return torch.cuda.LongTensor(*args)
    else:
        return torch.LongTensor(*args)


def GetTensor(tensor):
    if CUDA_ENABLED:
        return tensor.cuda()
    else:
        return tensor


def tensor(data, dtype):
    return torch.tensor(data, dtype=dtype, device=device())


def device():
    return "cuda:{}".format(torch.cuda.current_device()) if CUDA_ENABLED else "cpu"
