#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Callable, Dict, List, Tuple, Union

import torch
from caffe2.python import core
from caffe2.python.onnx.backend_rep import Caffe2Rep
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.config.field_config import FeatureConfig
from pytext.data import CommonMetadata
from pytext.fields import FieldMeta
from pytext.utils import onnx


class ModelExporter(Component):
    """
    Model exporter exports a PyTorch model to Caffe2 model using ONNX

    Attributes:
        input_names (List[Str]): names of the input variables to model forward
            function, in a flattened way.
            e.g: forward(tokens, dict) where tokens is List[Tensor] and dict is
            a tuple of value and length: (List[Tensor], List[Tensor]) the
            input names should looks like ['token', 'dict_value', 'dict_length']
        dummy_model_input (Tuple[torch.Tensor]): dummy values to define the
            shape of input tensors, should exactly match the shape of the model
            forward function
        vocab_map (Dict[str, List[str]]): dict of input feature names
            to corresponding index_to_string array, e.g:
            ::

                {
                    "text": ["<UNK>", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"],
                    "dict": ["<UNK>", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
                }
        output_names (List[Str]): names of output variables
    """

    __COMPONENT_TYPE__ = ComponentType.EXPORTER

    class Config(ConfigBase):
        export_logits: bool = False
        export_raw_to_metrics: bool = False

    @classmethod
    def from_config(
        cls,
        config,
        feature_config: FeatureConfig,
        target_config: Union[ConfigBase, List[ConfigBase]],
        meta: CommonMetadata,
        *args,
        **kwargs,
    ):
        """
            Gather all the necessary metadata from configs and global metadata to be
            used in exporter
        """
        input_names, dummy_model_input, vocab_map = cls.get_feature_metadata(
            feature_config, meta.features
        )
        if not isinstance(target_config, list):
            target_config = [target_config]
        output_names = [
            name for target in target_config for name in target.export_output_names
        ]
        return cls(config, input_names, dummy_model_input, vocab_map, output_names)

    @classmethod
    def get_feature_metadata(
        cls, feature_config: FeatureConfig, feature_meta: Dict[str, FieldMeta]
    ):
        # The number of names in input_names *must* be equal to the number of
        # tensors passed in dummy_input
        (
            input_names,
            dummy_model_input,
            feature_itos_map,
        ) = cls._get_exportable_metadata(
            lambda x: isinstance(x, ConfigBase), feature_config, feature_meta
        )
        cls._add_feature_lengths(input_names, dummy_model_input)
        return input_names, tuple(dummy_model_input), feature_itos_map

    def __init__(self, config, input_names, dummy_model_input, vocab_map, output_names):
        super().__init__(config)
        self.input_names = input_names
        self.output_names = output_names
        self.dummy_model_input = dummy_model_input
        self.vocab_map = vocab_map or {}
        # validate feature vocab
        for name in self.vocab_map:
            if name not in self.input_names:
                raise ValueError(
                    f"{name} is not found in input names {self.input_names}, \
                    there's a mismatch"
                )

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
        return onnx.add_feats_numericalize_ops(c2_prepared, self.vocab_map, input_names)

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
        model to append additional operators to caffe2 net

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
            workspace, init_net, predict_net, model_out, *output_names
        )

        # optionally include the last decoder layer of pytorch model
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

    def export_to_caffe2(
        self, model, export_path: str, export_onnx_path: str = None
    ) -> List[str]:
        """
        export pytorch model to caffe2 by first using ONNX to convert logic in forward
        function to a caffe2 net, and then prepend/append additional operators to
        the caffe2 net according to the model

        Args:
            model (Model): pytorch model to export
            export_path (str): path to save the exported caffe2 model
            export_onnx_path (str): path to save the exported onnx model

        Returns:
            final_output_names: list of caffe2 model output names
        """
        c2_prepared = onnx.pytorch_to_caffe2(
            model,
            self.dummy_model_input,
            self.input_names,
            self.output_names,
            export_path,
            export_onnx_path,
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
        onnx.export_nets_to_predictor_file(
            c2_prepared,
            final_input_names,
            final_out_names,
            export_path,
            self.get_extra_params(),
        )
        return final_out_names

    def export_to_metrics(self, model, metric_channels):
        """
        Exports the pytorch model to tensorboard as a graph.

        Args:
            model (Model): pytorch model to export
            metric_channels (List[Channel]): outputs of model's execution graph
        """

        for mc in metric_channels or []:
            if self.config.export_raw_to_metrics:
                mc.export(model, self.dummy_model_input, operator_export_type="RAW")
            else:
                mc.export(model, self.dummy_model_input)

    @classmethod
    def _get_exportable_metadata(
        cls,
        exportable_filter: Callable,
        feature_config: FeatureConfig,
        feature_meta: Dict[str, FieldMeta],
    ) -> Tuple[List[str], List, Dict]:
        # The number of names in input_names *must* be equal to the number of
        # tensors passed in dummy_input
        input_names: List[str] = []
        dummy_model_input: List = []
        feature_itos_map = {}

        for name, feat_config in feature_config._asdict().items():
            if exportable_filter(feat_config):
                input_names.extend(feat_config.export_input_names)
                if getattr(feature_meta[name], "vocab", None):
                    feature_itos_map[feat_config.export_input_names[0]] = feature_meta[
                        name
                    ].vocab.itos
                dummy_model_input.append(feature_meta[name].dummy_model_input)
        return input_names, dummy_model_input, feature_itos_map

    @classmethod
    def _add_feature_lengths(cls, input_names: List[str], dummy_model_input: List):
        """If any of the input_names have tokens or seq_tokens, add the length
        of those tokens to dummy_input
        """
        if "tokens_vals" in input_names:
            dummy_model_input.append(
                torch.tensor([1, 1], dtype=torch.long)
            )  # token lengths
            input_names.append("tokens_lens")
        if "seq_tokens_vals" in input_names:
            dummy_model_input.append(
                torch.tensor([1, 1], dtype=torch.long)
            )  # seq lengths
            input_names.append("seq_tokens_lens")
