#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


def stringify(token_indices, vocab):
    return " ".join([vocab[index] for index in token_indices])
