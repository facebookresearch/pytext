#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.torchscript.tensorizer.bert import ScriptBERTTensorizer
from pytext.torchscript.tensorizer.roberta import ScriptRoBERTaTensorizer


__all__ = ["ScriptBERTTensorizer", "ScriptRoBERTaTensorizer"]
