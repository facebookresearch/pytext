#!/usr/bin/env python3
from collections import OrderedDict
from typing import Any, List, NamedTupleMeta, Optional, Tuple, Union  # noqa


class InheritableNamedTupleMeta(NamedTupleMeta):
    def __new__(cls, typename, bases, ns):
        annotations = OrderedDict()
        annotations.update(ns.get("__annotations__", {}))
        for base in bases:
            if not issubclass(base, Tuple):
                continue
            base_fields = getattr(base, "__annotations__", {})

            for field_name in base_fields:
                # pass along default values
                if field_name in base._field_defaults and field_name not in ns:
                    ns[field_name] = base._field_defaults[field_name]
                if field_name not in annotations:
                    annotations[field_name] = base_fields[field_name]
                    if field_name not in ns:
                        annotations.move_to_end(field_name, last=False)
        if len(annotations) == 0:
            # fbl flow types don't support empty namedTuple,
            # add placeholder to workaround
            annotations["config_name_"] = str
            ns["config_name_"] = typename
        ns["__annotations__"] = annotations
        return super().__new__(cls, typename, bases, ns)
