#!/usr/bin/env python3


def cls_vars(cls):
    return [v for n, v in vars(cls).items() if not n.startswith("_")]