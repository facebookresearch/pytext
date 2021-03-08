#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reservedimport functools

import functools

# module decorator for specifying acceleration
# The purpose is to avoid ImportError when glow_decorator is not available
class accelerator:
    def __init__(self, specs, inputs_function=None):
        pass

    def __call__(self, module):
        @functools.wraps(module)
        def wrapper(*args, **kwargs):
            return module(*args, **kwargs)

        return wrapper

    @classmethod
    def _dfs_modules(cls, node, backend, results, submod_path=""):
        pass

    @classmethod
    def get_modules(cls, model, backend):
        pass

    @classmethod
    def get_module_from_path(cls, model, prefixes):
        pass

    @classmethod
    def get_embedding_module_from_path(cls, model, submod_path):
        pass
