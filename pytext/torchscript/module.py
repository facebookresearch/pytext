#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
from pytext.torchscript.tensorizer.tensorizer import ScriptTensorizer
from pytext.torchscript.utils import squeeze_1d, squeeze_2d


class ScriptTextModule(torch.jit.ScriptModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
    ):
        super().__init__()
        self.model = model
        self.output_layer = output_layer
        self.tensorizer = tensorizer

    @torch.jit.script_method
    def forward(self, texts: List[str]):
        input_tensors = self.tensorizer.tensorize(texts=squeeze_1d(texts))
        logits = self.model(input_tensors)
        return self.output_layer(logits)


class ScriptTokenModule(torch.jit.ScriptModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
    ):
        super().__init__()
        self.model = model
        self.output_layer = output_layer
        self.tensorizer = tensorizer

    @torch.jit.script_method
    def forward(self, tokens: List[List[str]]):
        input_tensors = self.tensorizer.tensorize(tokens=squeeze_2d(tokens))
        logits = self.model(input_tensors)
        return self.output_layer(logits)
