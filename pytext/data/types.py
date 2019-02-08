#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from pytext.config.component import Component, ComponentType


class DataType(Component):
    """
    A DataType is a structured python type that can be used as a rich type
    for building up model input tensors or target tensors, or used to compute
    metrics on directly.

    A `pytext.data.sources.DataSource` object takes some data set, regardless
    of the details of its storage or encoding, and yields values that have these
    of the details of its storage or encoding, and yields values that have these
    rich types.
    """

    __COMPONENT_TYPE__ = ComponentType.DATA_TYPE
    __EXPANSIBLE__ = True

    @classmethod
    def from_config(cls, config: Component.Config):
        return cls


class Label(DataType, str):
    """A string label for classification tasks."""


class Text(DataType, str):
    """Human language text."""
