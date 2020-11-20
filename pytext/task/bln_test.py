#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reservedimport functools

import functools

# compilation tuples is a list of tuples
# each tuple consists of a string identifying a backend
# and compiler options in dictionary format
def accelerator0(compilation_tuples):
    def decorator(func):
        @functools.wraps(func)
        def wrapper():
            print("hello")
            print("compilation tuples: ", compilation_tuples)
            func()

        return wrapper

    return decorator


# module decorator for specifying glow acceleration
class accelerator:
    modules = {}

    # compilation tuples is a list of tuples
    # each tuple consists of a string identifying a backend
    # and compiler options in dictionary format
    def __init__(self, compilation_tuples):
        self.compilation_tuples = compilation_tuples

    def __call__(self, module):
        # accelerator.modules.add(func.__name__)
        accelerator.modules[module.__name__] = self.compilation_tuples

        def wrapper(*args, **kwargs):
            return module(*args, **kwargs)

        return wrapper


@accelerator([("backend1", {"option1": "option"})])
def test(a, b):
    return a + b


print(test(7, 6))

print("modules added")
print(accelerator.modules["test"])


"""
class accelerator2:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.num_calls = 0

    def __call__(self, compilation_tuples):
        self.num_calls += 1
        print(f"Call {self.num_calls} ")
        return self.func()
"""


"""
class accelerator():

    # compilation tuples is a list of tuples
    # each tuple consists of a string identifying a backend
    # and compiler options in dictionary format
    def __init__(self, compilation_tuples):



    def decorator(func):
        @functools.wraps(func)
        def wrapper():
            print("hello")
            print("compilation tuples: ", compilation_tuples)
            func()

        return wrapper

    return decorator
"""

"""
@accelerator2(["backend1", {"option1": "flag"}])
def test():
    print("x")
"""

# test()


import accelerator_lowering
import torch
from torch import nn

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)

model = accelerator_lowering.AcceleratorTransformer(transformer_model)

print(model)
