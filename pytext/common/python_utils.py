#!/usr/bin/env python3
from collections import OrderedDict
from typing import Any, List, NamedTupleMeta, Optional, Tuple, Union  # noqa


def items(self):
    return self._asdict().items()


class InheritableNamedTupleMeta(NamedTupleMeta):
    def __new__(cls, typename, bases, ns):
        defaults = {}
        annotations = OrderedDict()
        ns["items"] = items
        for base in reversed(bases):
            if not issubclass(base, Tuple):
                continue
            defaults.update(base._field_defaults)
            annotations.update(getattr(base, "__annotations__", {}))
        defaults.update(ns)
        annotations.update(ns.get("__annotations__", {}))
        for field_name in defaults:
            if field_name in annotations:
                annotations.move_to_end(field_name)
        if not annotations:
            # fbl flow types don't support empty namedTuple,
            # add placeholder to workaround
            annotations["config_name_"] = str
            defaults["config_name_"] = typename
        defaults["__annotations__"] = annotations
        return super().__new__(cls, typename, bases, defaults)
