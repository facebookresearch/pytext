#!/usr/bin/env python3

from typing import Dict, List, Tuple

import torch
from caffe2.python import core
from caffe2.python.onnx.backend_rep import Caffe2Rep
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data import CommonMetadata
from pytext.utils import onnx_utils


class ModelExporter(Component):
    """Export the PyTorch model to Caffe2 model through ONNX (optional)"""

    __COMPONENT_TYPE__ = ComponentType.EXPORTER

    def __init__(self, config, input_names, output_names, dummy_model_input):
        """
        Model exporter exports a PyTorch model to Caffe2 model using ONNX

        Attributes:
            input_names (List[Str]): names of the input variables to model forward
                function, in a flattened way.
              e.g: forward(tokens, dict) where tokens is List[Tensor] and dict is
                   a tuple of value and length: (List[Tensor], List[Tensor]) the
                   input names should looks like ['token', 'dict_value', 'dict_length']
            output_names (List[Str]): names of output variables
            dummy_model_input (Tuple[torch.Tensor]): dummy values to define the
                shape of input tensors, should exactly match the shape of the model
                forward function
        """
        super().__init__(config)
        self.input_names = input_names
        self.output_names = output_names
        self.dummy_model_input = dummy_model_input

    def prepend_operators(
        self, c2_prepared: Caffe2Rep, input_names: List[str]
    ) -> Tuple[Caffe2Rep, List[str]]:
        """
        Prepend operators to the converted caffe2 net, do nothing by default

        Args:
            c2_prepared (Caffe2Rep): caffe2 net rep
            input_names (List[str]): current input names to the caffe2 net

        Returns:
            c2_prepared (Caffe2Rep): caffe2 net with prepended operators
            input_names (List[str]): list of input names for the new net
        """
        return c2_prepared, input_names

    def postprocess_output(
        self,
        init_net: core.Net,
        predict_net: core.Net,
        workspace: core.workspace,
        output_names: List[str],
        py_model,
    ):
        """
        Postprocess the model output, generate additional blobs for human readable
        prediction. By default it use export function of output layer from pytorch
        model to append addtional operators to caffe2 net

        Args:
            init_net (caffe2.python.Net): caffe2 init net created by the current graph
            predict_net (caffe2.python.Net): caffe2 net created by the current graph
            workspace (caffe2.python.workspace): caffe2 current workspace
            output_names (List[str]): current output names of the caffe2 net
            py_model (Model): original pytorch model object

        Returns:
            result: list of blobs that will be added to the caffe2 model
            final_output_names: list of output names of the blobs to add
        """
        model_out = py_model(*self.dummy_model_input)
        res = py_model.output_layer.export_to_caffe2(
            workspace,
            init_net,
            predict_net,
            model_out,
            *output_names,
            *self.label_names_list,
        )

        # optionaly include the last decoder layer of pytorch model
        final_output_names = [str(output) for output in res] + (
            output_names if self.config.export_logits else []
        )

        return res, final_output_names

    def get_extra_params(self) -> List[str]:
        """
        Returns:
            list of blobs to be added as extra params to the caffe2 model
        """
        return []

    def export_to_caffe2(self, model, export_path: str) -> List[str]:
        """
        export pytorch model to caffe2 by first using ONNX to convert logic in forward
        function to a caffe2 net, and then prepend/append addtional operators to
        the caffe2 net according to the model

        Args:
            model (Model): pytorch model to export
            export_path (str): path to save the exported caffe2 model

        Returns:
            final_output_names: list of caffe2 model output names
        """
        c2_prepared = onnx_utils.pytorch_to_caffe2(
            model,
            self.dummy_model_input,
            self.input_names,
            self.output_names,
            export_path,
        )
        c2_prepared, final_input_names = self.prepend_operators(
            c2_prepared, self.input_names
        )

        # Required because of https://github.com/pytorch/pytorch/pull/6456/files
        with c2_prepared.workspace._ctx:
            predict_net = core.Net(c2_prepared.predict_net)
            init_net = core.Net(c2_prepared.init_net)

            net_outputs, final_out_names = self.postprocess_output(
                init_net, predict_net, c2_prepared.workspace, self.output_names, model
            )
            for output in net_outputs:
                predict_net.AddExternalOutput(output)
            c2_prepared.predict_net = predict_net.Proto()
            c2_prepared.init_net = init_net.Proto()

        # Save predictor net to file
        onnx_utils.export_nets_to_predictor_file(
            c2_prepared,
            final_input_names,
            final_out_names,
            export_path,
            self.get_extra_params(),
        )
        return final_out_names


class TextModelExporter(ModelExporter):
    """
    Exporter for doc classificatoin and word tagging models

    Args:
        label_names_list: a list of output target class names e.g:
            [
                ["DOC1","DOC2","DOC3","DOC4"],
                ["WORD1","WORD2","WORD3","WORD4"]
            ]
        feature_itos_map: dict of input feature names to corresponding itos, e.g:
            {
                "text": ["<UNK>", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"],
                "dict": ["<UNK>", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
    """

    class Config(ConfigBase):
        export_logits: bool = False

    def __init__(
        self,
        config,
        label_names: List[List[str]],
        feature_itos_map: Dict[str, List[str]],
        meta: CommonMetadata,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.label_names_list = label_names
        self.vocab_map = feature_itos_map
        self.meta = meta
        # validate vocab_map
        for vocab_name in self.vocab_map:
            if vocab_name not in self.input_names:
                raise ValueError(
                    f"{vocab_name} is not found in input names {self.input_names}, \
                    there's a mismatch"
                )

    @classmethod
    def from_config(
        cls,
        config,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        meta: CommonMetadata,
        *args,
        **kwargs,
    ):
        # The number of names in input_names *must* be equal to the number of
        # tensors passed in dummy_input
        input_names = list(feature_config.word_feat.export_input_names)
        feature_itos_map = {
            feature_config.word_feat.export_input_names[0]: meta.features[
                DatasetFieldName.TEXT_FIELD
            ].vocab.itos
        }
        dummy_model_input = [torch.tensor([[1], [1]], dtype=torch.long)]  # tokens

        if feature_config.dict_feat:
            input_names.extend(feature_config.dict_feat.export_input_names)
            feature_itos_map[
                feature_config.dict_feat.export_input_names[0]
            ] = meta.features[DatasetFieldName.DICT_FIELD].vocab.itos
            dummy_model_input.append(
                (
                    torch.tensor([[2], [2]], dtype=torch.long),
                    torch.tensor([[1.5], [2.5]], dtype=torch.float),
                    torch.tensor([1, 1], dtype=torch.long),
                )
            )

        if feature_config.char_feat:
            input_names.extend(feature_config.char_feat.export_input_names)
            feature_itos_map[
                feature_config.char_feat.export_input_names[0]
            ] = meta.features[DatasetFieldName.CHAR_FIELD].vocab.itos
            dummy_model_input.append(
                torch.tensor([[[1, 1, 1]], [[1, 1, 1]]], dtype=torch.long)
            )

        dummy_model_input.append(
            torch.tensor([1, 1], dtype=torch.long)
        )  # token lengths
        input_names.append("tokens_lens")
        output_names: List[str] = []

        if getattr(label_config, "doc_label", None):
            output_names.extend(label_config.doc_label.export_output_names)

        if getattr(label_config, "word_label", None):
            output_names.extend(label_config.word_label.export_output_names)

        label_names = [label.vocab.itos for label in meta.labels.values()]

        return cls(
            config,
            label_names,
            feature_itos_map,
            meta,
            input_names,
            output_names,
            tuple(dummy_model_input),
        )

    def prepend_operators(
        self, c2_prepared: Caffe2Rep, input_names: List[str]
    ) -> Tuple[Caffe2Rep, List[str]]:
        return onnx_utils.add_feats_numericalize_ops(
            c2_prepared, self.vocab_map, input_names
        )
