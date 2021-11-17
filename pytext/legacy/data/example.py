#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
from functools import reduce


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    @classmethod
    def fromJSON(cls, data, fields):
        ex = cls()
        obj = json.loads(data)

        for key, vals in fields.items():
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]

                for val in vals:
                    # for processing the key likes 'foo.bar'
                    name, field = val
                    ks = key.split(".")

                    def reducer(obj, key):
                        if isinstance(obj, list):
                            results = []
                            for data in obj:
                                if key not in data:
                                    # key error
                                    raise ValueError(
                                        "Specified key {} was not found in "
                                        "the input data".format(key)
                                    )
                                else:
                                    results.append(data[key])
                            return results
                        else:
                            # key error
                            if key not in obj:
                                raise ValueError(
                                    "Specified key {} was not found in "
                                    "the input data".format(key)
                                )
                            else:
                                return obj[key]

                    v = reduce(reducer, ks, obj)
                    setattr(ex, name, field.preprocess(v))
        return ex

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                raise ValueError(
                    "Specified key {} was not found in " "the input data".format(key)
                )
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
        return ex

    @classmethod
    def fromCSV(cls, data, fields, field_to_index=None):
        if field_to_index is None:
            return cls.fromlist(data, fields)
        else:
            assert isinstance(fields, dict)
            data_dict = {f: data[idx] for f, idx in field_to_index.items()}
            return cls.fromdict(data_dict, fields)

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, str):
                    val = val.rstrip("\n")
                # Handle field tuples
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        setattr(ex, n, f.preprocess(val))
                else:
                    setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def fromtree(cls, data, fields, subtrees=False):
        try:
            from nltk.tree import Tree
        except ImportError:
            print(
                "Please install NLTK. "
                "See the docs at http://nltk.org for more information."
            )
            raise
        tree = Tree.fromstring(data)
        if subtrees:
            return [
                cls.fromlist([" ".join(t.leaves()), t.label()], fields)
                for t in tree.subtrees()
            ]
        return cls.fromlist([" ".join(tree.leaves()), tree.label()], fields)
