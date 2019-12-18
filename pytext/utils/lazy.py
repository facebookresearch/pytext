#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
from typing import Tuple

import torch
from torch import nn


class lazy_property(object):
    """
    More or less copy-pasta: http://stackoverflow.com/a/6849299
    Meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self._fget = fget
        self.__doc__ = fget.__doc__
        self.__name__ = fget.__name__

    def __get__(self, obj, obj_cls_type):
        if obj is None:
            return None
        value = self._fget(obj)
        setattr(obj, self.__name__, value)
        return value


class UninitializedLazyModuleError(Exception):
    """A lazy module was used improperly."""


class Infer:
    """A value which can be inferred from a forward pass. Infer objects should
    be passed as arguments or keyword arguments to Lazy objects; see Lazy
    documentation for more details.
    """

    def __init__(self, resolve_fn):
        """resolve_fn is called by Lazy on the arguments of the first forward pass
        to the Lazy module, and the Infer object will be replaced in the call by the
        output of this function. It should have the same signature as the
        Lazy-wrapped Module's forward function."""
        self.resolve = resolve_fn

    @classmethod
    def dimension(cls, dim):
        """A helper for creating Infer arguments looking at specific dimensions."""
        return cls(lambda input: input.size()[dim])


class Lazy(nn.Module):
    """
    A module which is able to infer some of its parameters from the inputs to
    its first forward pass. Lazy wraps any other nn.Module, and arguments can be passed
    that will be used to construct that wrapped Module after the first forward pass.
    If any of these arguments are Infer objects, those arguments will be replaced by
    calling the callback of the Infer object on the forward pass input.

    For instance,
    >>> Lazy(nn.Linear, Infer(lambda input: input.size(-1)), 4)
    Lazy()

    takes its in_features dimension from the last dimension of the input to its forward
    pass. This can be simplified to

    >>> Lazy(nn.Linear, Infer.dimension(-1), 4)

    or a partial can be created, for instance

    >>> LazyLinear = Lazy.partial(nn.Linear, Infer.dimension(-1))
    >>> LazyLinear(4)
    Lazy()

    Finally, these Lazy objects explicitly forbid treating themselves normally;
    they must instead be replaced by calling `init_lazy_modules`
    on your model before training. For instance,

    >>> ll = lazy.Linear(4)
    >>> seq = nn.Sequential(ll)
    >>> seq
    Sequential(
        0: Lazy(),
    )
    >>> init_lazy_modules(seq, torch.rand(1, 2)
    Sequential(
        0: Linear(in_features=2, out_features=4, bias=True)
    )
    """

    def __init__(self, module_class, *args, **kwargs):
        super().__init__()
        self._module = None
        self._module_class = module_class
        self._args = args
        self._kwargs = kwargs

    @classmethod
    def partial(cls, module_class, *args, **kwargs):
        return functools.partial(cls, module_class, *args, **kwargs)

    @property
    def _parameters(self):
        raise UninitializedLazyModuleError(
            "Must call init_lazy_modules before getting parameters"
        )

    @_parameters.setter
    def _parameters(self, value):
        return None

    def __setattr__(self, name, value):
        return object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        if not self._module:
            constructor_args = [
                (arg if not isinstance(arg, Infer) else arg.resolve(*args, **kwargs))
                for arg in self._args
            ]
            constructor_kwargs = {
                key: arg if not isinstance(arg, Infer) else arg.resolve(*args, **kwargs)
                for key, arg in self._kwargs.items()
            }
            self._module = self._module_class(*constructor_args, **constructor_kwargs)
        return self._module(*args, **kwargs)

    def resolve(self):
        """Must make a call to forward before calling this function; returns the
        full nn.Module object constructed using inferred arguments/dimensions."""
        if not self._module:
            raise UninitializedLazyModuleError(
                "Must call forward before calling resolve on a lazy module"
            )
        return self._module


def replace_lazy_modules(module):
    if isinstance(module, Lazy):
        module = module.resolve()
    children = list(module.named_children())
    for name, child in children:
        module.add_module(name, replace_lazy_modules(child))
    return module


def init_lazy_modules(
    module: nn.Module, dummy_input: Tuple[torch.Tensor, ...]
) -> nn.Module:
    """Finalize an nn.Module which has Lazy components. This will both mutate internal
    modules which have Lazy elements, and return a new non-lazy nn.Module (in case
    the top-level module itself is Lazy).

    Args:
        module: An nn.Module which may be lazy or contain Lazy subcomponents
        dummy_input: module is called with this input to ensure that Lazy subcomponents
            have been able to infer any parameters they need
    Returns:
        The full nn.Module object constructed using inferred arguments/dimensions.
    """
    module(*dummy_input)
    return replace_lazy_modules(module)


Linear = Lazy.partial(nn.Linear, Infer.dimension(-1))
LayerNorm = Lazy.partial(nn.LayerNorm, Infer.dimension(-1))
Conv1d = Lazy.partial(nn.Conv1d, Infer.dimension(-2))
