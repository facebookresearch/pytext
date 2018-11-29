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


def xaviervar(*size):
    t = torch.Tensor(*size)
    t = torch.nn.init.xavier_normal_(t)
    return Variable(t)


def FloatTensor(*args):
    if CUDA_ENABLED:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


def GetTensor(tensor):
    if CUDA_ENABLED:
        return tensor.cuda()
    else:
        return tensor
