#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.tensorizer.tensorizer import ScriptTensorizer
from pytext.torchscript.utils import ScriptInputType, squeeze_1d, squeeze_2d
from torch.nn import functional as F


def get_script_module_cls(input_type: ScriptInputType) -> torch.jit.ScriptModule:
    if input_type.is_text():
        return ScriptTextModule
    elif input_type.is_token():
        return ScriptTokenModule
    else:
        raise RuntimeError("Only support text or token input type...")


class ScriptModule(torch.jit.ScriptModule):
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
    def set_device(self, device: str):
        self.tensorizer.set_device(device)


class ScriptTextModule(ScriptModule):
    @torch.jit.script_method
    def forward(self, texts: List[str]):
        input_tensors = self.tensorizer(texts=squeeze_1d(texts))
        logits = self.model(input_tensors)
        return self.output_layer(logits)


class ScriptTokenModule(ScriptModule):
    @torch.jit.script_method
    def forward(self, tokens: List[List[str]]):
        input_tensors = self.tensorizer(pre_tokenized=squeeze_2d(tokens))
        logits = self.model(input_tensors)
        return self.output_layer(logits)


class ScriptTokenLanguageModule(ScriptModule):
    @torch.jit.script_method
    def forward(self, tokens: List[List[str]], languages: Optional[List[str]] = None):
        input_tensors = self.tensorizer(
            pre_tokenized=squeeze_2d(tokens), languages=squeeze_1d(languages)
        )
        logits = self.model(input_tensors)
        return self.output_layer(logits)


class ScriptTokenLanguageModuleWithDenseFeature(ScriptModule):
    @torch.jit.script_method
    def forward(
        self,
        tokens: List[List[str]],
        dense_feat: List[List[float]],
        languages: Optional[List[str]] = None,
    ):
        input_tensors = self.tensorizer(
            pre_tokenized=squeeze_2d(tokens), languages=squeeze_1d(languages)
        )
        logits = self.model(input_tensors, torch.tensor(dense_feat).float())
        return self.output_layer(logits)


class ScriptTextSquadModule(ScriptModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        max_ans_len: int,
    ):
        super().__init__(model, output_layer, tensorizer)
        self.max_ans_len = torch.jit.Attribute(max_ans_len, int)

    @torch.jit.script_method
    def forward(self, document: str, question: str) -> Tuple[int, int, float, float]:
        (
            tokens,
            pad_mask,
            segment_labels,
            positions,
            offset,
            start_indices,
            end_indices,
        ) = self.tensorizer(q_d_texts=[[question, document]])

        # use the traced model to get the span logits
        # {1, tokens_dim}
        start_pos_logits, end_pos_logits, has_ans_logits, _, _ = self.model(
            tokens, pad_mask, segment_labels, positions
        )

        # pass the logits through the scripted output layer
        ans_start_pos, ans_end_pos = self.output_layer(
            torch.zeros(0), start_pos_logits, end_pos_logits, self.max_ans_len
        )

        # Calculate the confidence score of the answer as log probability.
        # logP(ans) = logP(start) + logP(end)
        start_pos_score = (
            F.log_softmax(start_pos_logits, 1)
            .gather(1, ans_start_pos.view(-1, 1))
            .squeeze(-1)
        )
        end_pos_score = (
            F.log_softmax(end_pos_logits, 1)
            .gather(1, ans_end_pos.view(-1, 1))
            .squeeze(-1)
        )
        ans_score = start_pos_score + end_pos_score

        # Calculate the log probability of the question being answerable.
        has_ans_score = F.log_softmax(has_ans_logits, dim=1).squeeze()[1].item()

        # detokenize answer span [token span to character span]
        return (
            # - q_len to get the right document token index
            start_indices[0, ans_start_pos - offset].item(),
            end_indices[0, ans_end_pos - offset].item(),
            ans_score.item(),
            has_ans_score,
        )
