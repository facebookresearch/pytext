#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytext.config import ExportConfig
from pytext.torchscript.batchutils import (
    input_size,
    limit_list,
    clip_list,
    limit_listlist,
    clip_listlist,
    limit_listlist_float,
    clip_listlist_float,
    destructure_tensor,
    destructure_tensor_list,
    destructure_any_list,
    zip_batch_any_list_list,
    zip_batch_tensor_list,
    make_batch_texts_dense,
    make_prediction_texts,
    make_prediction_texts_dense,
    max_tokens,
    nonify_listlist_float,
    validate_dense_feat,
)
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


def deprecation_warning(export_conf: ExportConfig):
    if (
        (export_conf.inference_interface is not None)
        or (export_conf.accelerate is not None)
        or (export_conf.seq_padding_control is not None)
        or (export_conf.batch_padding_control is not None)
    ):
        msg = [
            "***********  DEPRECATION WARNING  **********",
            "Modules concurrently supporting untokenized",
            "and tokenized inputs are being deprecated!",
            "",
            "Preferably, use the corresponding Pytext{Type}Module",
            "hierarchy (sans 'Script') classes to offer models",
            "including tokenization.",
            "*********************************************",
        ]
        for line in msg:
            print(line)


class ScriptPyTextEmbeddingModule(torch.jit.ScriptModule):
    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer):

        super().__init__()
        self.model = model
        self.tensorizer = tensorizer
        self.argno = -1
        log_class_usage(self.__class__)

    def validate(self, export_conf: ExportConfig):
        deprecation_warning(export_conf)

    @torch.jit.script_method
    def set_device(self, device: str):
        self.tensorizer.set_device(device)

    @torch.jit.script_method
    def get_max_seq_len(self) -> int:
        """
        This function returns the maximum sequence length for the model,
        if it is defined, otherwise raises a Runtime Error.
        """
        if hasattr(self.tensorizer, "max_seq_len"):
            if self.tensorizer.max_seq_len is not None:
                return self.tensorizer.max_seq_len

        raise RuntimeError("max_seq_len not defined")

    @torch.jit.script_method
    def get_max_batch_len(self) -> int:
        """
        This function returns the maximum batch length for the model,
        if it is defined, otherwise -1.
        """
        if hasattr(self.tensorizer, "batch_padding_control"):
            batch_padding_control = self.tensorizer.batch_padding_control
            if batch_padding_control is not None:
                return batch_padding_control[-1]

        return -1

    @torch.jit.script_method
    def set_padding_control(self, dimension: str, control: Optional[List[int]]):
        """
        This functions will be called to set a padding style.
        None - No padding
        List: first element 0, round seq length to the smallest list element larger than inputs
        """
        self.tensorizer.set_padding_control(dimension, control)

    @torch.jit.script_method
    def uses_dense_feat(self) -> bool:
        return False

    @torch.jit.script_method
    def forward_validate_dense_feat(
        self,
        dense_feat: Optional[List[List[float]]],
    ) -> List[List[float]]:
        if self.uses_dense_feat():
            if dense_feat is None:
                raise RuntimeError(
                    "Dense feature (dense_feat) is required for this model type, but not present."
                )
            else:
                return dense_feat
        else:
            if dense_feat is not None:
                raise RuntimeError(
                    "Dense feature (dense_feat) not allowed for this model type"
                )
            else:
                return []

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        return self.model(input_tensors).cpu()

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        # self.uses_dense_feat() indicates use: False
        dense_feat: Optional[List[List[float]]] = None,
    ) -> torch.Tensor:
        self.forward_validate_dense_feat(dense_feat)

        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        return self._forward(inputs)

    @torch.jit.script_method
    def forward(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        # self.uses_dense_feat() indicates use: False
        dense_feat: Optional[List[List[float]]] = None,
    ):  # returns torch.Tensor or List[Any]

        self.forward_validate_dense_feat(dense_feat)

        input_len = input_size(texts, multi_texts, tokens)
        max_batch = self.get_max_batch_len()
        if max_batch <= 0:
            max_batch = input_len

        result = self.forward_impl(
            limit_list(texts, max_batch),
            limit_listlist(multi_texts, max_batch),
            limit_listlist(tokens, max_batch),
            limit_list(languages, max_batch),
            limit_listlist_float(dense_feat, max_batch),
        )

        if input_len > max_batch:
            texts = clip_list(texts, max_batch)
            multi_texts = clip_listlist(multi_texts, max_batch)
            tokens = clip_listlist(tokens, max_batch)
            languages = clip_list(languages, max_batch)
            dense_feat = clip_listlist_float(dense_feat, max_batch)

            while input_size(texts, multi_texts, tokens) > 0:
                result_extension = self.forward_impl(
                    limit_list(texts, max_batch),
                    limit_listlist(multi_texts, max_batch),
                    limit_listlist(tokens, max_batch),
                    limit_list(languages, max_batch),
                    limit_listlist_float(dense_feat, max_batch),
                )

                # the result of forward is either a torch.Tensor or a List[Any]
                if isinstance(result, torch.Tensor):
                    result = torch.cat([result, result_extension], dim=0)
                else:
                    result.extend(result_extension)

                # prepare next iteration
                texts = clip_list(texts, max_batch)
                multi_texts = clip_listlist(multi_texts, max_batch)
                tokens = clip_listlist(tokens, max_batch)
                languages = clip_list(languages, max_batch)
                dense_feat = clip_listlist_float(dense_feat, max_batch)

        if isinstance(result, torch.Tensor):
            torch._assert(
                input_len == result.size()[0],
                "Tensor output size must match input size",
            )
        else:
            torch._assert(
                input_len == len(result), "List output size must match input size"
            )

        return result

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                Optional[List[str]],  # texts
                Optional[List[List[str]]],  # multi_texts
                Optional[List[List[str]]],  # tokens
                Optional[List[str]],  # languages
                Optional[List[List[float]]],  # dense_feat must be None
            ]
        ],
    ):  # List[torch.Tensor] or List[List[Any]]

        batchsize = len(batch)

        client_batch_texts: List[int] = []
        client_batch_tokens: List[int] = []
        zip_batch_list: List[int] = []

        flat_texts: List[str] = []
        flat_tokens: List[List[str]] = []
        flat_dense_feat_texts: List[List[float]] = []
        flat_dense_feat_tokens: List[List[float]] = []

        for i in range(batchsize):
            batch_element_texts = batch[i][0]
            batch_element_tokens = batch[i][2]
            batch_element_dense_feat = batch[i][4]

            if batch_element_texts is not None:
                flat_texts.extend(batch_element_texts)
                client_batch_texts.append(len(batch_element_texts))
                flat_dense_feat_texts.extend(
                    validate_dense_feat(
                        batch_element_dense_feat,
                        len(batch_element_texts),
                        self.uses_dense_feat(),
                    )
                )
                zip_batch_list.append(1)
            elif batch_element_tokens is not None:
                flat_tokens.extend(batch_element_tokens)
                client_batch_tokens.append(len(batch_element_tokens))
                flat_dense_feat_tokens.extend(
                    validate_dense_feat(
                        batch_element_dense_feat,
                        len(batch_element_tokens),
                        self.uses_dense_feat(),
                    )
                )
                zip_batch_list.append(-1)
            else:
                # At present, we abort the entire batch if
                # any batch element is malformed.
                #
                # Possible refinement:
                # we can skip malformed requests,
                # and return a list plus an indiction that one or more
                # batch elements (and which ones) were malformed
                raise RuntimeError("Malformed request.")

        if len(flat_texts) == 0 and len(flat_tokens) == 0:
            raise RuntimeError("This is not good. Empty request batch.")

        if len(flat_texts) > 0 and len(flat_tokens) > 0:
            raise RuntimeError("Mixing tokens and texts not supported in this service.")
            # flat_result_texts = self.forward(
            #     texts=flat_texts[:max_batch],
            #     multi_texts=None,
            #     tokens=None,
            #     languages=None,
            #     dense_feat=nonify_listlist_float(flat_dense_feat_texts),
            # )
            # flat_result_tokens = self.forward(
            #     texts=None,
            #     multi_texts=None,
            #     tokens=flat_tokens[:max_batch],
            #     languages=None,
            #     dense_feat=nonify_listlist_float(flat_dense_feat_tokens),
            # )
        elif len(flat_texts) > 0:
            flat_result_texts = self.forward(
                texts=flat_texts,
                multi_texts=None,
                tokens=None,
                languages=None,
                dense_feat=nonify_listlist_float(flat_dense_feat_texts),
            )
            # ignored in logic, this makes type system happy
            flat_result_tokens = flat_result_texts
        else:  #  len(flat_tokens) > 0:
            flat_result_tokens = self.forward(
                texts=None,
                multi_texts=None,
                tokens=flat_tokens,
                languages=None,
                dense_feat=nonify_listlist_float(flat_dense_feat_tokens),
            )
            # ignored in logic, this makes type system happy
            flat_result_texts = flat_result_tokens

        # if torch.jit.isinstance(flat_result_tokens, torch.Tensor):
        if isinstance(flat_result_tokens, torch.Tensor):
            # destructure flat result tensor combining
            #   cross-request batches and client side
            #   batches into a cross-request list of
            #   client-side batch tensors
            return zip_batch_tensor_list(
                zip_batch_list,
                destructure_tensor(client_batch_texts, flat_result_texts),
                destructure_tensor(client_batch_tokens, flat_result_tokens),
            )
        else:
            # destructure result list of any result type combining
            #   cross-request batches and client side
            #   batches into a cross-request list of
            #   client-side result lists
            result_texts_any_list: List[Any] = torch.jit.annotate(List[Any], [])
            for v in flat_result_texts:
                result_texts_any_list.append(v)

            result_tokens_any_list: List[Any] = torch.jit.annotate(List[Any], [])
            for v in flat_result_tokens:
                result_tokens_any_list.append(v)

            return zip_batch_any_list_list(
                zip_batch_list,
                destructure_any_list(client_batch_texts, result_texts_any_list),
                destructure_any_list(client_batch_tokens, result_tokens_any_list),
            )

    @torch.jit.script_method
    def make_batch(
        self,
        mega_batch: List[
            Tuple[
                Optional[List[str]],  # texts
                Optional[List[List[str]]],  # multi_texts
                Optional[List[List[str]]],  # tokens
                Optional[List[str]],  # languages
                Optional[List[List[float]]],  # dense_feat must be None
                int,
            ]
        ],
        goals: Dict[str, str],
    ) -> List[
        List[
            Tuple[
                Optional[List[str]],  # texts
                Optional[List[List[str]]],  # multi_texts
                Optional[List[List[str]]],  # tokens
                Optional[List[str]],  # languages
                Optional[List[List[float]]],  # dense_feat must be None
                int,
            ]
        ]
    ]:
        # The next lines sort all cross-request batch elements by the token length.
        # Note that cross-request batch element can in turn be a client batch.
        mega_batch_key_list = [
            (max_tokens(self.tensorizer.tokenize(x[0], x[2])), n)
            for (n, x) in enumerate(mega_batch)
        ]
        sorted_mega_batch_key_list = sorted(mega_batch_key_list)
        sorted_mega_batch = [mega_batch[n] for (key, n) in sorted_mega_batch_key_list]

        # TBD: allow model server to specify batch size in goals dictionary
        max_bs: int = 10
        len_mb = len(mega_batch)
        num_batches = (len_mb + max_bs - 1) // max_bs

        batch_list: List[
            List[
                Tuple[
                    Optional[List[str]],  # texts
                    Optional[List[List[str]]],  # multi_texts
                    Optional[List[List[str]]],  # tokens
                    Optional[List[str]],  # language,
                    Optional[List[List[float]]],  # dense_feat must be None
                    int,  # position
                ]
            ]
        ] = []

        start = 0

        for _i in range(num_batches):
            end = min(start + max_bs, len_mb)
            batch_list.append(sorted_mega_batch[start:end])
            start = end

        return batch_list


class ScriptPyTextEmbeddingModuleIndex(ScriptPyTextEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        index: int = 0,
    ):
        super().__init__(model, tensorizer)
        self.index: int = index
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def uses_dense_feat(self) -> bool:
        return False

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        return self.model(input_tensors)[self.index].cpu()


class ScriptPyTextModule(ScriptPyTextEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
    ):
        super().__init__(model, tensorizer)
        # A PyText Module is an EmbeddingModule with an output layer
        self.output_layer = output_layer

    @torch.jit.script_method
    def uses_dense_feat(self) -> bool:
        return False

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        # self.uses_dense_feat() indicates use: False
        dense_feat: Optional[List[List[float]]] = None,
    ):
        self.forward_validate_dense_feat(dense_feat)

        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        input_tensors = self.tensorizer(inputs)
        logits = self.model(input_tensors)
        return self.output_layer(logits)


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
    def uses_dense_feat(self) -> bool:
        return True

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput, dense_tensor: torch.Tensor):
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        return self.model(input_tensors, dense_tensor).cpu()

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        # self.uses_dense_feat() indicates use: True
        dense_feat: Optional[List[List[float]]] = None,
    ) -> torch.Tensor:
        dense_feat = self.forward_validate_dense_feat(dense_feat)

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


class ScriptPyTextModuleWithDense(ScriptPyTextEmbeddingModuleWithDense):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        normalizer: VectorNormalizer,
    ):
        super().__init__(model, tensorizer, normalizer)
        self.output_layer = output_layer
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def uses_dense_feat(self) -> bool:
        return True

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        # self.uses_dense_feat() indicates use: True
        dense_feat: Optional[List[List[float]]] = None,
    ):
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        input_tensors = self.tensorizer(inputs)

        if dense_feat is not None:
            dense_feat = self.normalizer.normalize(dense_feat)
            dense_tensor = torch.tensor(dense_feat, dtype=torch.float)
        else:
            raise RuntimeError("dense feature cannot be None.")

        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        logits = self.model(input_tensors, dense_tensor)
        return self.output_layer(logits)


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
    def uses_dense_feat(self) -> bool:
        return True

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
    def uses_dense_feat(self) -> bool:
        return False

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        reps, seq_lens = self.model(input_tensors)
        reps = reps.cpu()
        seq_lens = seq_lens.cpu()
        return [reps[i, : seq_lens[i]] for i in range(len(seq_lens))]

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: Optional[List[str]] = None,
        # multi_texts is of shape [batch_size, num_columns]
        multi_texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        # self.uses_dense_feat() indicates use: False
        dense_feat: Optional[List[List[float]]] = None,
    ) -> List[torch.Tensor]:
        self.forward_validate_dense_feat(dense_feat)

        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, multi_texts),
            tokens=squeeze_2d(tokens),
            languages=squeeze_1d(languages),
        )
        return self._forward(inputs)


######################## Two Tower ################################
class ScriptTwoTowerModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def set_device(self, device: str):
        self.right_tensorizer.set_device(device)
        self.left_tensorizer.set_device(device)

    @torch.jit.script_method
    def set_padding_control(self, dimension: str, control: Optional[List[int]]):
        """
        This functions will be called to set a padding style.
        None - No padding
        List: first element 0, round seq length to the smallest list element larger than inputs
        """
        self.right_tensorizer.set_padding_control(dimension, control)
        self.left_tensorizer.set_padding_control(dimension, control)

    def validate(self, export_conf: ExportConfig):
        deprecation_warning(export_conf)


class ScriptPyTextTwoTowerModule(ScriptTwoTowerModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
    ):
        super().__init__()
        self.model = model
        self.output_layer = output_layer
        self.right_tensorizer = right_tensorizer
        self.left_tensorizer = left_tensorizer

    @torch.jit.script_method
    def forward(
        self,
        right_texts: Optional[List[str]] = None,
        left_texts: Optional[List[str]] = None,
        right_tokens: Optional[List[List[str]]] = None,
        left_tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
    ):
        right_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(right_texts),
            tokens=squeeze_2d(right_tokens),
            languages=squeeze_1d(languages),
        )
        right_input_tensors = self.right_tensorizer(right_inputs)
        left_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(left_texts),
            tokens=squeeze_2d(left_tokens),
            languages=squeeze_1d(languages),
        )
        left_input_tensors = self.left_tensorizer(left_inputs)
        logits = self.model(right_input_tensors, left_input_tensors)
        return self.output_layer(logits)


class ScriptPyTextTwoTowerModuleWithDense(ScriptPyTextTwoTowerModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
        right_normalizer: VectorNormalizer,
        left_normalizer: VectorNormalizer,
    ):
        super().__init__(model, output_layer, right_tensorizer, left_tensorizer)
        self.right_normalizer = right_normalizer
        self.left_normalizer = left_normalizer

    @torch.jit.script_method
    def forward(
        self,
        right_dense_feat: List[List[float]],
        left_dense_feat: List[List[float]],
        right_texts: Optional[List[str]] = None,
        left_texts: Optional[List[str]] = None,
        right_tokens: Optional[List[List[str]]] = None,
        left_tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
    ):
        right_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(right_texts),
            tokens=squeeze_2d(right_tokens),
            languages=squeeze_1d(languages),
        )
        right_input_tensors = self.right_tensorizer(right_inputs)
        left_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(left_texts),
            tokens=squeeze_2d(left_tokens),
            languages=squeeze_1d(languages),
        )
        left_input_tensors = self.left_tensorizer(left_inputs)

        right_dense_feat = self.right_normalizer.normalize(right_dense_feat)
        left_dense_feat = self.left_normalizer.normalize(left_dense_feat)
        right_dense_tensor = torch.tensor(right_dense_feat, dtype=torch.float)
        left_dense_tensor = torch.tensor(left_dense_feat, dtype=torch.float)
        if self.right_tensorizer.device != "":
            right_dense_tensor = right_dense_tensor.to(self.right_tensorizer.device)
        if self.left_tensorizer.device != "":
            left_dense_tensor = left_dense_tensor.to(self.left_tensorizer.device)

        logits = self.model(
            right_input_tensors,
            left_input_tensors,
            right_dense_tensor,
            left_dense_tensor,
        )
        return self.output_layer(logits)


class ScriptPyTextTwoTowerEmbeddingModule(ScriptTwoTowerModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
    ):
        super().__init__()
        self.model = model
        self.right_tensorizer = right_tensorizer
        self.left_tensorizer = left_tensorizer
        # We only support texts input for TwoTower until further updates
        self.argno = 0
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def _forward(self, right_inputs: ScriptBatchInput, left_inputs: ScriptBatchInput):
        right_input_tensors = self.right_tensorizer(right_inputs)
        left_input_tensors = self.left_tensorizer(left_inputs)

        return self.model(right_input_tensors, left_input_tensors).cpu()

    @torch.jit.script_method
    def forward(
        self,
        right_texts: Optional[List[str]] = None,
        left_texts: Optional[List[str]] = None,
        right_tokens: Optional[List[List[str]]] = None,
        left_tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
    ) -> torch.Tensor:
        right_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(right_texts),
            tokens=squeeze_2d(right_tokens),
            languages=squeeze_1d(languages),
        )
        left_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(left_texts),
            tokens=squeeze_2d(left_tokens),
            languages=squeeze_1d(languages),
        )
        return self._forward(right_inputs, left_inputs)

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                Optional[List[str]],  # right_texts
                Optional[List[str]],  # left_texts
                Optional[List[List[str]]],  # right_tokens
                Optional[List[List[str]]],  # left_tokens
                Optional[List[str]],  # languages
                Optional[List[List[float]]],  # right_dense_feat
                Optional[List[List[float]]],  # left_dense_feat
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
        # res_list: List[torch.Tensor] = []

        if argno == TEXTS:
            flat_right_texts: List[str] = []
            flat_left_texts: List[str] = []

            for i in range(batchsize):
                batch_right_element = batch[i][0]
                batch_left_element = batch[i][1]
                if batch_right_element is not None:
                    flat_right_texts.extend(batch_right_element)
                    client_batch.append(len(batch_right_element))
                else:
                    # At present, we abort the entire batch if
                    # any batch element is malformed.
                    #
                    # Possible refinement:
                    # we can skip malformed requests,
                    # and return a list plus an indiction that one or more
                    # batch elements (and which ones) were malformed
                    raise RuntimeError("Malformed request.")

                if batch_left_element is not None:
                    flat_left_texts.extend(batch_left_element)
                else:
                    raise RuntimeError("Malformed request.")

            flat_result: torch.Tensor = self.forward(
                right_texts=flat_right_texts,
                left_texts=flat_left_texts,
                right_tokens=None,
                left_tokens=None,
                languages=None,
                right_dense_feat=None,
                left_dense_feat=None,
            )

        else:
            raise RuntimeError("Parameter type unsupported")

        # destructure flat result tensor combining
        #   cross-request batches and client side
        #   batches into a cross-request list of
        #   client-side batch tensors
        return destructure_tensor(client_batch, flat_result)

    @torch.jit.script_method
    def make_batch(
        self,
        mega_batch: List[
            Tuple[
                Optional[List[str]],  # right_texts
                Optional[List[str]],  # left_texts
                Optional[List[List[str]]],  # right_tokens
                Optional[List[List[str]]],  # left_tokens
                Optional[List[str]],  # languages
                Optional[List[List[float]]],  # right_dense_feat
                Optional[List[List[float]]],  # left_dense_feat
                int,
            ]
        ],
        goals: Dict[str, str],
    ) -> List[
        List[
            Tuple[
                Optional[List[str]],  # right_texts
                Optional[List[str]],  # left_texts
                Optional[List[List[str]]],  # right_tokens
                Optional[List[List[str]]],  # left_tokens
                Optional[List[str]],  # languages
                Optional[List[List[float]]],  # right_dense_feat
                Optional[List[List[float]]],  # left_dense_feat
                int,
            ]
        ]
    ]:

        argno = self.argno

        if argno == -1:
            raise RuntimeError("Argument number not specified during export.")

        # The next lines sort all cross-request batch elements by the token length of right_.
        # Note that cross-request batch element can in turn be a client batch.
        mega_batch_key_list = [
            (max_tokens(self.right_tensorizer.tokenize(x[0], x[2])), n)
            for (n, x) in enumerate(mega_batch)
        ]
        sorted_mega_batch_key_list = sorted(mega_batch_key_list)
        sorted_mega_batch = [mega_batch[n] for (key, n) in sorted_mega_batch_key_list]

        # TBD: allow model server to specify batch size in goals dictionary
        max_bs: int = 10
        len_mb = len(mega_batch)
        num_batches = (len_mb + max_bs - 1) // max_bs

        batch_list: List[
            List[
                Tuple[
                    Optional[List[str]],  # right_texts
                    Optional[List[str]],  # left_texts
                    Optional[List[List[str]]],  # right_tokens
                    Optional[List[List[str]]],  # left_tokens
                    Optional[List[str]],  # languages
                    Optional[List[List[float]]],  # right_dense_feat
                    Optional[List[List[float]]],  # left_dense_feat
                    int,  # position
                ]
            ]
        ] = []

        start = 0

        for _i in range(num_batches):
            end = min(start + max_bs, len_mb)
            batch_list.append(sorted_mega_batch[start:end])
            start = end

        return batch_list


class ScriptPyTextTwoTowerEmbeddingModuleWithDense(ScriptPyTextTwoTowerEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
        right_normalizer: VectorNormalizer,
        left_normalizer: VectorNormalizer,
    ):
        super().__init__(model, right_tensorizer, left_tensorizer)
        self.right_normalizer = right_normalizer
        self.left_normalizer = left_normalizer
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def _forward(
        self,
        right_inputs: ScriptBatchInput,
        left_inputs: ScriptBatchInput,
        right_dense_tensor: torch.Tensor,
        left_dense_tensor: torch.Tensor,
    ):
        right_input_tensors = self.right_tensorizer(right_inputs)
        left_input_tensors = self.left_tensorizer(left_inputs)

        if self.right_tensorizer.device != "":
            right_dense_tensor = right_dense_tensor.to(self.right_tensorizer.device)
        if self.left_tensorizer.device != "":
            left_dense_tensor = left_dense_tensor.to(self.left_tensorizer.device)

        return self.model(
            right_input_tensors,
            left_input_tensors,
            right_dense_tensor,
            left_dense_tensor,
        ).cpu()

    @torch.jit.script_method
    def forward(
        self,
        right_texts: Optional[List[str]] = None,
        left_texts: Optional[List[str]] = None,
        right_tokens: Optional[List[List[str]]] = None,
        left_tokens: Optional[List[List[str]]] = None,
        languages: Optional[List[str]] = None,
        right_dense_feat: Optional[List[List[float]]] = None,
        left_dense_feat: Optional[List[List[float]]] = None,
    ) -> torch.Tensor:
        if right_dense_feat is None or left_dense_feat is None:
            raise RuntimeError("Expect dense feature.")

        right_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(right_texts),
            tokens=squeeze_2d(right_tokens),
            languages=squeeze_1d(languages),
        )
        left_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(left_texts),
            tokens=squeeze_2d(left_tokens),
            languages=squeeze_1d(languages),
        )

        right_dense_feat = self.right_normalizer.normalize(right_dense_feat)
        left_dense_feat = self.left_normalizer.normalize(left_dense_feat)
        right_dense_tensor = torch.tensor(right_dense_feat, dtype=torch.float)
        left_dense_tensor = torch.tensor(left_dense_feat, dtype=torch.float)

        sentence_embedding = self._forward(
            right_inputs, left_inputs, right_dense_tensor, left_dense_tensor
        )
        return sentence_embedding


############################################################################
#
# New module hierarchy Pytext* mirrors ScriptPytext* while reflecting
# advances in pytext models:
#  * Integrated tokenization - sole interface is texts which will be tokenized
#  * Multi-lingual models - no need to specify languages
#
# All new modules provide:
#  * Cross-request batching support
#  * Batch optimization support
#  * Sequence length and batch size padding for accelerators
#
#
# The inputs and outputs for cross-request batching with make_prediction
# are described in this post:
#   https://fb.workplace.com/groups/401165540538639/permalink/556111271710731/
#
# The inputs and outputs for batch optimization with make_batch
# are described in this post:
#   https://fb.workplace.com/groups/401165540538639/permalink/607830233205501/
#

#############################################################
# Pytext Classes:


class PyTextEmbeddingModule(torch.jit.ScriptModule):
    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer):
        super().__init__()
        self.model = model
        self.tensorizer = tensorizer
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def set_device(self, device: str):
        self.tensorizer.set_device(device)

    @torch.jit.script_method
    def set_padding_control(self, dimension: str, control: Optional[List[int]]):
        """
        This functions will be called to set a padding style.
        None - No padding
        List: first element 0, round seq length to the smallest list element larger than inputs
        """
        self.tensorizer.set_padding_control(dimension, control)

    @torch.jit.script_method
    def get_max_seq_len(self) -> int:
        """
        This function returns the maximum sequence length for the model,
        if it is defined, otherwise None.
        """
        if hasattr(self.tensorizer, "max_seq_len"):
            if self.tensorizer.max_seq_len is not None:
                return self.tensorizer.max_seq_len

        raise RuntimeError("max_seq_len not defined")

    @torch.jit.script_method
    def get_max_batch_len(self) -> int:
        """
        This function returns the maximum batch length for the model,
        if it is defined, otherwise -1.
        """
        if hasattr(self.tensorizer, "batch_padding_control"):
            batch_padding_control = self.tensorizer.batch_padding_control
            if batch_padding_control is not None:
                return batch_padding_control[-1]

        return -1

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        return self.model(input_tensors).cpu()

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, None),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        return self._forward(inputs)

    @torch.jit.script_method
    def forward(
        self,
        texts: List[str],
    ) -> torch.Tensor:

        input_len = len(texts)
        max_batch = self.get_max_batch_len()
        if max_batch < 0:
            max_batch = input_len

        result = self.forward_impl(
            texts[:max_batch],
        )

        if input_len > max_batch:
            texts = texts[max_batch:]

            while len(texts) > 0:
                result_extension = self.forward_impl(
                    texts[:max_batch],
                )
                # the result of forward is either a torch.Tensor or a List[Any]
                if isinstance(result, torch.Tensor):
                    result = torch.cat([result, result_extension], dim=0)
                else:
                    result.extend(result_extension)

                texts = texts[max_batch:]

        return result

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                List[str],  # texts
            ]
        ],
    ) -> List[torch.Tensor]:

        flat_result: torch.Tensor = self.forward(
            texts=make_prediction_texts(batch),
        )

        return destructure_tensor([len(be[0]) for be in batch], flat_result)

    @torch.jit.script_method
    def make_batch(
        self,
        mega_batch: List[
            Tuple[
                List[str],  # texts
                int,
            ]
        ],
        goals: Dict[str, str],
    ) -> List[List[Tuple[List[str], int,]]]:  # texts

        # The next lines sort all cross-request batch elements by the token length.
        # Note that cross-request batch element can in turn be a client batch.
        mega_batch_key_list = [
            (max_tokens(self.tensorizer.tokenize(x[0], None)), n)
            for (n, x) in enumerate(mega_batch)
        ]
        sorted_mega_batch_key_list = sorted(mega_batch_key_list)
        sorted_mega_batch = [mega_batch[n] for (_, n) in sorted_mega_batch_key_list]

        # TBD: allow model server to specify batch size in goals dictionary
        max_bs: int = 10
        len_mb = len(mega_batch)
        num_batches = (len_mb + max_bs - 1) // max_bs

        batch_list: List[
            List[
                Tuple[
                    List[str],  # texts
                    int,  # position
                ]
            ]
        ] = []

        start = 0

        for _i in range(num_batches):
            end = min(start + max_bs, len_mb)
            batch_list.append(sorted_mega_batch[start:end])
            start = end

        return batch_list


# PytextLayerModule is a PytextEmbeddingModule with an additional output layer


class PyTextLayerModule(PyTextEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
    ):
        super().__init__(model, tensorizer)
        self.output_layer = output_layer

    @torch.jit.script_method
    def forward_impl(self, texts: List[str]):
        # logits = super().forward(texts)
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, None),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        input_tensors = self.tensorizer(inputs)
        logits = self.model(input_tensors)
        # </> logits = super().forward(texts)
        return self.output_layer(logits)


# PytextEmbeddingModuleIndex is a PytextEmbeddingModule with an additional Index


class PyTextEmbeddingModuleIndex(PyTextEmbeddingModule):
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


# PytextEmbeddingModuleWithDense is a PytextEmbeddingModule with an additional dense_feat


class PyTextEmbeddingModuleWithDense(PyTextEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        normalizer: VectorNormalizer,
        concat_dense: bool = False,
    ):
        super().__init__(model, tensorizer)
        self.normalizer = normalizer
        self.concat_dense: bool = concat_dense
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput, dense_tensor: torch.Tensor):
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        return self.model(input_tensors, dense_tensor).cpu()

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: List[str],
        dense_feat: List[List[float]],
    ) -> torch.Tensor:

        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, None),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        # call model
        dense_feat = self.normalizer.normalize(dense_feat)
        dense_tensor = torch.tensor(dense_feat, dtype=torch.float)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)

        sentence_embedding = self._forward(inputs, dense_tensor)
        if self.concat_dense:
            return torch.cat([sentence_embedding, dense_tensor], 1)
        else:
            return sentence_embedding

    @torch.jit.script_method
    def forward(
        self,
        texts: List[str],
        dense_feat: List[List[float]],
    ) -> torch.Tensor:

        input_len = len(texts)
        max_batch = self.get_max_batch_len()
        if max_batch < 0:
            max_batch = input_len

        result = self.forward_impl(texts[:max_batch], dense_feat[:max_batch])

        if input_len > max_batch:
            texts = texts[max_batch:]
            dense_feat = dense_feat[max_batch:]

            while len(texts) > 0:
                result_extension = self.forward_impl(
                    texts[:max_batch], dense_feat[:max_batch]
                )
                # the result of forward is either a torch.Tensor or a List[Any]
                if isinstance(result, torch.Tensor):
                    result = torch.cat([result, result_extension], dim=0)
                else:
                    result.extend(result_extension)

                texts = texts[max_batch:]
                dense_feat = dense_feat[max_batch:]

        return result

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                List[str],  # texts
                List[List[float]],  # dense
            ]
        ],
    ) -> List[torch.Tensor]:

        flat_texts, flat_dense = make_prediction_texts_dense(batch)

        flat_result: torch.Tensor = self.forward(
            texts=flat_texts,
            dense_feat=flat_dense,
        )

        return destructure_tensor([len(be[0]) for be in batch], flat_result)

    @torch.jit.script_method
    def make_batch(
        self,
        mega_batch: List[
            Tuple[
                List[str],  # texts
                List[List[float]],  # dense
                int,
            ]
        ],
        goals: Dict[str, str],
    ) -> List[List[Tuple[List[str], List[List[float]], int,]]]:  # texts  # dense

        return make_batch_texts_dense(self.tensorizer, mega_batch, goals)


# PytextLayerModuleWithDense is a PytextEmbeddingModuleWithDense with an additional output layer


class PyTextLayerModuleWithDense(PyTextEmbeddingModuleWithDense):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptTensorizer,
        normalizer: VectorNormalizer,
    ):
        super().__init__(model, tensorizer, normalizer)
        self.output_layer = output_layer
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def forward_impl(
        self,
        texts: List[str],
        dense_feat: List[List[float]],
    ):
        # logits = super().forward(texts, dense_feat)
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, None),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        input_tensors = self.tensorizer(inputs)
        dense_feat = self.normalizer.normalize(dense_feat)

        dense_tensor = torch.tensor(dense_feat, dtype=torch.float)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        logits = self.model(input_tensors, dense_tensor)
        # </>logits = super().forward(texts, dense_feat)
        return self.output_layer(logits)


# PytextEmbeddingModuleWithDenseIndex is a PytextEmbeddingModuleWithDense with an additional Index


class PyTextEmbeddingModuleWithDenseIndex(PyTextEmbeddingModuleWithDense):
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
        # return super()._forward(inputs, dense_tensor)[self.index].cpu()
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != "":
            dense_tensor = dense_tensor.to(self.tensorizer.device)
        return self.model(input_tensors, dense_tensor)[self.index].cpu()
        # </> return super()._forward(inputs, dense_tensor)[self.index].cpu()


class PyTextVariableSizeEmbeddingModule(PyTextEmbeddingModule):
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
    def forward_impl(self, texts: List[str]) -> List[torch.Tensor]:
        inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(texts, None),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        return self._forward(inputs)

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                List[str],  # texts
            ]
        ],
    ) -> List[List[torch.Tensor]]:

        flat_result: List[torch.Tensor] = self.forward(
            texts=make_prediction_texts(batch),
        )

        return destructure_tensor_list([len(be[0]) for be in batch], flat_result)


#############################################################
# PytextTwoTower Classes:
#
#  mirrors the inheritance order of Pytext modules.
#  *** please keep order and inheritance structure        ***
#  *** in sync between these two hierarchies              ***
#


class PyTextTwoTowerEmbeddingModule(torch.jit.ScriptModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
    ):
        super().__init__()
        self.model = model
        self.right_tensorizer = right_tensorizer
        self.left_tensorizer = left_tensorizer
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def set_device(self, device: str):
        self.right_tensorizer.set_device(device)
        self.left_tensorizer.set_device(device)

    @torch.jit.script_method
    def set_padding_control(self, dimension: str, control: Optional[List[int]]):
        """
        This functions will be called to set a padding style.
        None - No padding
        List: first element 0, round seq length to the smallest list element larger than inputs
        """
        self.right_tensorizer.set_padding_control(dimension, control)
        self.left_tensorizer.set_padding_control(dimension, control)

    @torch.jit.script_method
    def _forward(self, right_inputs: ScriptBatchInput, left_inputs: ScriptBatchInput):
        right_input_tensors = self.right_tensorizer(right_inputs)
        left_input_tensors = self.left_tensorizer(left_inputs)

        return self.model(right_input_tensors, left_input_tensors).cpu()

    @torch.jit.script_method
    def forward(
        self,
        right_texts: List[str],
        left_texts: List[str],
    ) -> torch.Tensor:
        right_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(right_texts),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        left_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(left_texts),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        return self._forward(right_inputs, left_inputs)

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                List[str],  # right_texts
                List[str],  # left_texts
            ]
        ],
    ) -> List[torch.Tensor]:

        batchsize = len(batch)

        flat_right_texts: List[str] = []
        flat_left_texts: List[str] = []

        for i in range(batchsize):
            batch_right_element = batch[i][0]
            batch_left_element = batch[i][1]

            flat_right_texts.extend(batch_right_element)
            flat_left_texts.extend(batch_left_element)

        flat_result: torch.Tensor = self.forward(
            right_texts=flat_right_texts,
            left_texts=flat_left_texts,
        )

        return destructure_tensor([len(be[0]) for be in batch], flat_result)

    @torch.jit.script_method
    def make_batch(
        self,
        mega_batch: List[
            Tuple[
                List[str],  # right_texts
                List[str],  # left_texts
                int,
            ]
        ],
        goals: Dict[str, str],
    ) -> List[List[Tuple[List[str], List[str], int,]]]:  # right_texts  # left_texts

        # The next lines sort all cross-request batch elements by the token length of right_.
        # Note that cross-request batch element can in turn be a client batch.
        mega_batch_key_list = [
            (max_tokens(self.right_tensorizer.tokenize(x[0], None)), n)
            for (n, x) in enumerate(mega_batch)
        ]
        sorted_mega_batch_key_list = sorted(mega_batch_key_list)
        sorted_mega_batch = [mega_batch[n] for (key, n) in sorted_mega_batch_key_list]

        # TBD: allow model server to specify batch size in goals dictionary
        max_bs: int = 10
        len_mb = len(mega_batch)
        num_batches = (len_mb + max_bs - 1) // max_bs

        batch_list: List[
            List[
                Tuple[
                    List[str],  # right_texts
                    List[str],  # left_texts
                    int,  # position
                ]
            ]
        ] = []

        start = 0

        for _i in range(num_batches):
            end = min(start + max_bs, len_mb)
            batch_list.append(sorted_mega_batch[start:end])
            start = end

        return batch_list


class PyTextTwoTowerLayerModule(PyTextTwoTowerEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
    ):
        super().__init__(model, right_tensorizer, left_tensorizer)
        self.output_layer = output_layer

    @torch.jit.script_method
    def forward(
        self,
        right_texts: List[str],
        left_texts: List[str],
    ):
        logits = super().forward(right_texts, left_texts)
        return self.output_layer(logits)


class PyTextTwoTowerEmbeddingModuleWithDense(PyTextTwoTowerEmbeddingModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
        right_normalizer: VectorNormalizer,
        left_normalizer: VectorNormalizer,
    ):
        super().__init__(model, right_tensorizer, left_tensorizer)
        self.right_normalizer = right_normalizer
        self.left_normalizer = left_normalizer
        log_class_usage(self.__class__)

    @torch.jit.script_method
    def _forward(
        self,
        right_inputs: ScriptBatchInput,
        left_inputs: ScriptBatchInput,
        right_dense_tensor: torch.Tensor,
        left_dense_tensor: torch.Tensor,
    ):
        right_input_tensors = self.right_tensorizer(right_inputs)
        left_input_tensors = self.left_tensorizer(left_inputs)

        if self.right_tensorizer.device != "":
            right_dense_tensor = right_dense_tensor.to(self.right_tensorizer.device)
        if self.left_tensorizer.device != "":
            left_dense_tensor = left_dense_tensor.to(self.left_tensorizer.device)

        return self.model(
            right_input_tensors,
            left_input_tensors,
            right_dense_tensor,
            left_dense_tensor,
        ).cpu()

    @torch.jit.script_method
    def forward(
        self,
        right_texts: List[str],
        left_texts: List[str],
        right_dense_feat: List[List[float]],
        left_dense_feat: List[List[float]],
    ) -> torch.Tensor:

        right_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(right_texts),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        left_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(left_texts),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )

        right_dense_feat = self.right_normalizer.normalize(right_dense_feat)
        left_dense_feat = self.left_normalizer.normalize(left_dense_feat)
        right_dense_tensor = torch.tensor(right_dense_feat, dtype=torch.float)
        left_dense_tensor = torch.tensor(left_dense_feat, dtype=torch.float)

        sentence_embedding = self._forward(
            right_inputs, left_inputs, right_dense_tensor, left_dense_tensor
        )
        return sentence_embedding


class PyTextTwoTowerLayerModuleWithDense(PyTextTwoTowerLayerModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        right_tensorizer: ScriptTensorizer,
        left_tensorizer: ScriptTensorizer,
        right_normalizer: VectorNormalizer,
        left_normalizer: VectorNormalizer,
    ):
        super().__init__(model, output_layer, right_tensorizer, left_tensorizer)
        self.right_normalizer = right_normalizer
        self.left_normalizer = left_normalizer

    @torch.jit.script_method
    def forward(
        self,
        right_texts: List[str],
        left_texts: List[str],
        right_dense_feat: List[List[float]],
        left_dense_feat: List[List[float]],
    ):
        right_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(right_texts),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        right_input_tensors = self.right_tensorizer(right_inputs)
        left_inputs: ScriptBatchInput = ScriptBatchInput(
            texts=resolve_texts(left_texts),
            tokens=squeeze_2d(None),
            languages=squeeze_1d(None),
        )
        left_input_tensors = self.left_tensorizer(left_inputs)

        right_dense_feat = self.right_normalizer.normalize(right_dense_feat)
        left_dense_feat = self.left_normalizer.normalize(left_dense_feat)
        right_dense_tensor = torch.tensor(right_dense_feat, dtype=torch.float)
        left_dense_tensor = torch.tensor(left_dense_feat, dtype=torch.float)
        if self.right_tensorizer.device != "":
            right_dense_tensor = right_dense_tensor.to(self.right_tensorizer.device)
        if self.left_tensorizer.device != "":
            left_dense_tensor = left_dense_tensor.to(self.left_tensorizer.device)

        logits = self.model(
            right_input_tensors,
            left_input_tensors,
            right_dense_tensor,
            left_dense_tensor,
        )
        return self.output_layer(logits)

    @torch.jit.script_method
    def make_prediction(
        self,
        batch: List[
            Tuple[
                List[str],  # right_texts
                List[str],  # left_texts
                List[List[float]],  # right_dense_feat
                List[List[float]],  # left_dense_feat
            ]
        ],
    ) -> List[torch.Tensor]:

        batchsize = len(batch)

        flat_right_texts: List[str] = []
        flat_left_texts: List[str] = []
        flat_right_dense: List[List[float]] = []
        flat_left_dense: List[List[float]] = []

        for i in range(batchsize):
            batch_right_element = batch[i][0]
            batch_left_element = batch[i][1]
            batch_right_dense_element = batch[i][2]
            batch_left_dense_element = batch[i][3]

            flat_right_texts.extend(batch_right_element)
            flat_left_texts.extend(batch_left_element)
            flat_right_dense.extend(batch_right_dense_element)
            flat_left_dense.extend(batch_left_dense_element)

        flat_result: torch.Tensor = self.forward(
            right_texts=flat_right_texts,
            left_texts=flat_left_texts,
            right_dense_feat=flat_right_dense,
            left_dense_feat=flat_left_dense,
        )

        return destructure_tensor([len(be[0]) for be in batch], flat_result)

    @torch.jit.script_method
    def make_batch(
        self,
        mega_batch: List[
            Tuple[
                List[str],  # right_texts
                List[str],  # left_texts
                List[List[float]],  # right_dense_feat
                List[List[float]],  # left_dense_feat
                int,
            ]
        ],
        goals: Dict[str, str],
    ) -> List[
        List[
            Tuple[
                List[str],  # right_texts
                List[str],  # left_texts
                List[List[float]],  # right_dense_feat
                List[List[float]],  # left_dense_feat
                int,
            ]
        ]
    ]:

        # The next lines sort all cross-request batch elements by the token length of right_.
        # Note that cross-request batch element can in turn be a client batch.
        mega_batch_key_list = [
            (max_tokens(self.right_tensorizer.tokenize(x[0], None)), n)
            for (n, x) in enumerate(mega_batch)
        ]
        sorted_mega_batch_key_list = sorted(mega_batch_key_list)
        sorted_mega_batch = [mega_batch[n] for (key, n) in sorted_mega_batch_key_list]

        # TBD: allow model server to specify batch size in goals dictionary
        max_bs: int = 10
        len_mb = len(mega_batch)
        num_batches = (len_mb + max_bs - 1) // max_bs

        batch_list: List[
            List[
                Tuple[
                    List[str],  # right_texts
                    List[str],  # left_texts
                    List[List[float]],  # right_dense_feat
                    List[List[float]],  # left_dense_feat
                    int,  # position
                ]
            ]
        ] = []

        start = 0

        for _i in range(num_batches):
            end = min(start + max_bs, len_mb)
            batch_list.append(sorted_mega_batch[start:end])
            start = end

        return batch_list
