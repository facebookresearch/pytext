#!/usr/bin/env python3

from typing import Dict, List, Tuple

import torch
from pytext.common.registry import EXPORTER, component
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.utils import onnx_utils
from caffe2.python import core
from caffe2.python.onnx.backend_rep import Caffe2Rep


class ModelExporter:
    """Export the PyTorch model to Caffe2 model through ONNX (optional)"""

    def __init__(self, input_names, output_names, dummy_model_input):
        """Define the names and shapes of input/output
        1 input names, names of the input variables to model forward function,
          in a flattened way
          e.g: forward(tokens, dict) where tokens is List[Tensor] and dict is
               a tuple of value and length: (List[Tensor], List[Tensor]) the
               input names should looks like ['token', 'dict_value', 'dict_length']
        2 output names, names of output variables
        3 dummy_model_input, dummy values to define the shape of input tensors,
          should exactly match the shape of the model forward function
        """
        self.input_names = input_names
        self.output_names = output_names
        self.dummy_model_input = dummy_model_input

    def prepend_operators(
        self, c2_prepared: Caffe2Rep, input_names: List[str]
    ) -> Tuple[Caffe2Rep, List[str]]:
        """Prepend operators to the converted caffe2 net, do nothing by default
            args:
            c2_prepared: caffe2 net rep
            input_names: current input names to the caffe2 net
            returns:
            caffe2 net with prepended operators
            list of input names for the new net
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
        """Postprocess the model output, generate additional blobs for human readable
            result.
            args:
            init_net: caffe2 init net created by the current graph
            predict_net: caffe2 net created by the current graph
            workspace: caffe2 current workspace
            output_names: current output names of the caffe2 net
            py_model: original pytorch model object
            returns:
            list of blobs that will be added to the caffe2 model
            list of output names of the blobs to add
        """
        return [], output_names

    def export_to_caffe2(self, model, export_path: str) -> List[str]:
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
            c2_prepared, final_input_names, final_out_names, export_path
        )
        return final_out_names


class TextModelExporterConfig(ConfigBase):
    pass


@component(EXPORTER, config_cls=TextModelExporterConfig)
class TextModelExporter(ModelExporter):
    """Exporter for doc classifier and word tagger models
        args:
        class_names_list: a list of output target class names e.g:
            [
                ["DOC1","DOC2","DOC3","DOC4"],
                ["WORD1","WORD2","WORD3","WORD4"]
            ]
        vocab_map: dict of input feature names to corresponding itos, e.g:
            {
                "text": ["<UNK>", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"],
                "dict": ["<UNK>", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
            }
        score_axis_list: list of axis of score in the output tensor. e.g:
            doc model has output as [batch_size, score] and word model has output
            as [batch_size, words, score], so input can be [1 , 2]
    """

    def __init__(
        self,
        class_names: List[List[str]],
        feature_itos_map: Dict[str, List[str]],
        score_axis_list: List[int],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_names_list = class_names
        self.score_axis_list = score_axis_list
        self.vocab_map = feature_itos_map
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
        config: TextModelExporterConfig,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        class_names: List[List[str]],
        feature_itos_map: Dict[str, List[str]],
        **kwargs,
    ):
        input_names = list(feature_config.word_feat.export_input_names)
        if feature_config.dict_feat:
            input_names.extend(feature_config.dict_feat.export_input_names)

        output_names: List[str] = []
        axis: List[int] = []

        if label_config.doc_label:
            output_names.extend(label_config.doc_label.export_output_names)
            axis.append(1)

        if label_config.word_label:
            output_names.extend(label_config.word_label.export_output_names)
            axis.append(2)

        # The sample input should exactly match the shape of the model forward function
        dummy_model_input = (
            torch.tensor([[1], [1]], dtype=torch.long),
            torch.tensor([1, 1], dtype=torch.long),
            (
                torch.tensor([[2], [2]], dtype=torch.long),
                torch.tensor([[1.5], [2.5]], dtype=torch.float),
                torch.tensor([1, 1], dtype=torch.long),
            ),
        )

        return cls(
            class_names,
            feature_itos_map,
            axis,
            input_names,
            output_names,
            dummy_model_input,
        )

    def prepend_operators(
        self, c2_prepared: Caffe2Rep, input_names: List[str]
    ) -> Tuple[Caffe2Rep, List[str]]:
        return onnx_utils.add_feats_numericalize_ops(
            c2_prepared, self.vocab_map, input_names
        )

    def postprocess_output(
        self,
        init_net: core.Net,
        predict_net: core.Net,
        workspace: core.workspace,
        output_names: List[str],
        py_model,
    ) -> Tuple[List[core.BlobReference], List[str]]:
        res = []
        for class_names, output_score, axis in zip(
            self.class_names_list, output_names, self.score_axis_list
        ):
            if hasattr(py_model, "crf") and py_model.crf and axis == 2:
                tmp_out_score = py_model.crf.export_to_caffe2(
                    workspace, init_net, predict_net, output_score
                )
            else:
                softmax_out = predict_net.Softmax(output_score, axis=axis)
                tmp_out_score = predict_net.Log(softmax_out)
            label_scores = predict_net.Split(tmp_out_score, class_names, axis=axis)

            # Make sure label_scores is iterable
            if not isinstance(label_scores, tuple):
                label_scores = (label_scores,)
            for name, label_score in zip(class_names, label_scores):
                res.append(
                    predict_net.Copy(label_score, "{}:{}".format(output_score, name))
                )

        return res, [str(output) for output in res]
