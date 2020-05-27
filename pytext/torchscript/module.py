#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

import torch
from pytext.torchscript.tensorizer.normalizer import VectorNormalizer
from pytext.torchscript.tensorizer.tensorizer import ScriptTensorizer
from pytext.torchscript.utils import ScriptBatchInput, squeeze_1d, squeeze_2d


@torch.jit.script
def resolve_texts(
    texts: Optional[List[str]] = None, multi_texts: Optional[List[List[str]]] = None
) -> Optional[List[List[str]]]:
    if texts is not None:
        return squeeze_1d(texts)
    return multi_texts


class ScriptModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def set_device(self, device: str):
        self.tensorizer.set_device(device)


class ScriptPyTextModule(ScriptModule):
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
    def forward(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
    ):
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        input_tensors = self.tensorizer(inputs)
        logits = self.model(input_tensors)
        return self.output_layer(logits)


class ScriptPyTextModuleWithDense(ScriptPyTextModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        normalizer: VectorNormalizer,
    ):
        super().__init__(model, output_layer, tensorizer)
        self.normalizer = normalizer

    @torch.jit.script_method
    def forward(
        self,
        dense_feat: List[List[float]],
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
    ):
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        input_tensors = self.tensorizer(inputs)
        dense_feat = self.normalizer.normalize(dense_feat)

        dense_tensor = torch.tensor(dense_feat, dtype=torch.float)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        logits = self.model(input_tensors, dense_tensor)
        return self.output_layer(logits)


class ScriptPyTextEmbeddingModule(ScriptModule):
    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer):
        super().__init__()
        self.model = model
        self.tensorizer = tensorizer

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        return self.model(input_tensors).cpu()

    @torch.jit.script_method
    def forward(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        dense_feat: Optional[List[List[float]]] = None,
    ) -> torch.Tensor:
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        return self._forward(inputs)


class ScriptPyTextEmbeddingModuleIndex(ScriptPyTextEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        index: int = 0,
    ):
        super().__init__(model, tensorizer)
        self.index = torch.jit.Attribute(index, int)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        return self.model(input_tensors)[self.index].cpu()


class ScriptPyTextEmbeddingModuleWithDense(ScriptPyTextEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        normalizer: VectorNormalizer,
        concat_dense: bool = False,
    ):
        super().__init__(model, tensorizer)
        self.normalizer = normalizer
        self.concat_dense = torch.jit.Attribute(concat_dense, bool)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput, dense_tensor: torch.Tensor):
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        return self.model(input_tensors, dense_tensor).cpu()

    @torch.jit.script_method
    def forward(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        dense_feat: Optional[List[List[float]]] = None,
    ) -> torch.Tensor:
        if dense_feat is None:
            raise RuntimeError("Expect dense feature.")

        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        # call model
        dense_feat = self.normalizer.normalize(dense_feat)
        dense_tensor = torch.tensor(dense_feat, dtype=torch.float)

        sentence_embedding = self._forward(inputs, dense_tensor)
        if self.concat_dense:
            return torch.cat([sentence_embedding, dense_tensor], 1)
        else:
            return sentence_embedding


class ScriptPyTextEmbeddingModuleWithDenseIndex(ScriptPyTextEmbeddingModuleWithDense):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        normalizer: VectorNormalizer,
        index: int = 0,
        concat_dense: bool = True,
    ):
        super().__init__(model, tensorizer, normalizer, concat_dense)
        self.index = torch.jit.Attribute(index, int)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput, dense_tensor: torch.Tensor):
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        return self.model(input_tensors, dense_tensor)[self.index].cpu()
