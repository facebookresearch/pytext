#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional, Tuple

import torch
from pytext.torchscript.tensorizer.normalizer import VectorNormalizer
from pytext.torchscript.tensorizer.tensorizer import ScriptTensorizer
from pytext.torchscript.utils import ScriptBatchInput, squeeze_1d, squeeze_2d
from pytext.utils.usage import log_class_usage


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
        log_class_usage(self.__class__)

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


class ScriptPyTextTwoTowerModuleWithDense(ScriptPyTextModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        right_normalizer: VectorNormalizer,
        left_normalizer: VectorNormalizer,
    ):
        super().__init__(model, output_layer, tensorizer)
        self.right_normalizer = right_normalizer
        self.left_normalizer = left_normalizer

    @torch.jit.script_method
    def forward(
        self,
        right_dense_feat: List[List[float]],
        left_dense_feat: List[List[float]],
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
        right_dense_feat = self.right_normalizer.normalize(right_dense_feat)
        left_dense_feat = self.left_normalizer.normalize(left_dense_feat)

        right_dense_tensor = torch.tensor(right_dense_feat, dtype=torch.float)
        left_dense_tensor = torch.tensor(left_dense_feat, dtype=torch.float)
        if self.tensorizer.device != "":
            right_dense_tensor = right_dense_tensor.to(self.tensorizer.device)
            left_dense_tensor = left_dense_tensor.to(self.tensorizer.device)
        logits = self.model(input_tensors, right_dense_tensor, left_dense_tensor)
        return self.output_layer(logits)


class ScriptPyTextEmbeddingModule(ScriptModule):
    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer):
        super().__init__()
        self.model = model
        self.tensorizer = tensorizer
        self.argno = -1
        log_class_usage(self.__class__)

    def inference_interface(self, argument_type: str):

        # Argument types and Tuple indices
        TEXTS = 0
        # MULTI_TEXTS = 1
        # TOKENS = 2
        # LANGUAGES = 3
        # DENSE_FEAT = 4

        if argument_type == "texts":
            self.argno = TEXTS
        else:
            raise RuntimeError("Unsupported argument type.")

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

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                Optional[List[str]],  # texts
                Optional[List[List[str]]],  # multi_texts
                Optional[List[List[str]]],  # tokens
                Optional[List[str]],  # languages
                Optional[List[List[float]]],  # dense_feat
            ]
        ],
    ) -> List[torch.Tensor]:

        argno = self.argno

        if argno == -1:
            raise RuntimeError("Argument number not specified during export.")

        batchsize = len(batch)

        # Argument types and Tuple indices
        TEXTS = 0
        # MULTI_TEXTS = 1
        # TOKENS = 2
        # LANGUAGES = 3
        # DENSE_FEAT = 4

        client_batch: List[int] = []
        res_list: List[torch.Tensor] = []

        if argno == TEXTS:
            flat_texts: List[str] = []

            for i in range(batchsize):
                batch_element = batch[i][0]
                if batch_element is not None:
                    flat_texts.extend(batch_element)
                    client_batch.append(len(batch_element))
                else:
                    # At present, we abort the entire batch if
                    # any batch element is malformed.
                    #
                    # Possible refinement:
                    # we can skip malformed requests,
                    # and return a list plus an indiction that one or more
                    # batch elements (and which ones) were malformed
                    raise RuntimeError("Malformed request.")

            flat_result = self.forward(
                texts=flat_texts,
                multi_texts=None,
                tokens=None,
                languages=None,
                dense_feat=None,
            )

        else:
            raise RuntimeError("Parameter type unsupported")

        # destructure flat result list combining
        #   cross-request batches and client side
        #   batches into a cross-request list of
        #  client-side batch result lists
        start = 0
        for elems in client_batch:
            end = start + elems
            res_list.append(flat_result.narrow(0, start, elems))
            start = end

        return res_list


class ScriptPyTextEmbeddingModuleIndex(ScriptPyTextEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        index: int = 0,
    ):
        super().__init__(model, tensorizer)
        self.index = torch.jit.Attribute(index, int)
        log_class_usage(self.__class__)

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
        log_class_usage(self.__class__)

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
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput, dense_tensor: torch.Tensor):
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        return self.model(input_tensors, dense_tensor)[self.index].cpu()


class ScriptPyTextVariableSizeEmbeddingModule(ScriptPyTextEmbeddingModule):
    """
    Assumes model returns a tuple of representations and sequence lengths, then slices
    each example's representation according to length. Returns a list of tensors. The
    slicing is easier to do outside a traced model.
    """

    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer):
        super().__init__(model, tensorizer)
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        reps, seq_lens = self.model(input_tensors)
        reps = reps.cpu()
        seq_lens = seq_lens.cpu()
        return [reps[i, : seq_lens[i]] for i in range(len(seq_lens))]

    @torch.jit.script_method
    def forward(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        dense_feat: Optional[List[List[float]]] = None,
    ) -> List[torch.Tensor]:
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        return self._forward(inputs)
