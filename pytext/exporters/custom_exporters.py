#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Dict, List

from caffe2.python import core, workspace
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, FloatVectorConfig
from pytext.config.module_config import ExporterType
from pytext.exporters.exporter import ModelExporter
from pytext.fields import FieldMeta
from pytext.utils import mobile_onnx
from pytext.utils.file_io import PathManager


def save_caffe2_pb_net(path, model):
    with PathManager.open(path, "wb") as f:
        f.write(model.SerializeToString())


class DenseFeatureExporter(ModelExporter):
    """
    Exporter for models that have DenseFeatures as input to the decoder
    """

    @classmethod
    def get_feature_metadata(
        cls, feature_config: FeatureConfig, feature_meta: Dict[str, FieldMeta]
    ):
        # add all features EXCEPT dense features. The features exported here
        # go through the representation layer
        (
            input_names_rep,
            dummy_model_input_rep,
            feature_itos_map_rep,
        ) = cls._get_exportable_metadata(
            lambda x: isinstance(x, ConfigBase)
            and not isinstance(x, FloatVectorConfig),
            feature_config,
            feature_meta,
        )

        # need feature lengths only for non-dense features
        cls._add_feature_lengths(input_names_rep, dummy_model_input_rep)

        # add dense features. These features don't go through the representation
        # layer, instead they go directly to the decoder
        (
            input_names_dense,
            dummy_model_input_dense,
            feature_itos_map_dense,
        ) = cls._get_exportable_metadata(
            lambda x: isinstance(x, FloatVectorConfig), feature_config, feature_meta
        )

        feature_itos_map_rep.update(feature_itos_map_dense)
        return (
            input_names_rep + input_names_dense,
            tuple(dummy_model_input_rep + dummy_model_input_dense),
            feature_itos_map_rep,
        )


class InitPredictNetExporter(ModelExporter):
    """
    Exporter for converting models to their caffe2 init and predict nets.
    Does not rely on c2_prepared, but rather splits the ONNX model into
    the init and predict nets directly.

    """

    def prepend_operators(self, init_net, predict_net, input_names: List[str]):
        return mobile_onnx.add_feats_numericalize_ops(
            init_net, predict_net, self.vocab_map, input_names
        )

    def postprocess_output(
        self, init_net, predict_net, workspace, output_names: List[str], model
    ):
        model_out = model(*self.dummy_model_input)
        res = model.output_layer.export_to_caffe2(
            workspace, init_net, predict_net, model_out, *output_names
        )
        final_output_names = [str(output) for output in res]
        return (res, final_output_names)

    def get_export_paths(self, path):
        export_dir = os.path.dirname(path)
        return (
            os.path.join(export_dir, "init_net.pb"),
            os.path.join(export_dir, "predict_net.pb"),
        )

    def export_to_caffe2(
        self, model, export_path: str, export_onnx_path: str = None
    ) -> List[str]:

        init_net_path, predict_net_path = self.get_export_paths(export_path)
        print(f"Saving caffe2 init net to {init_net_path}")
        print(f"Saving caffe2 init net to {predict_net_path}")

        init_net, predict_net = mobile_onnx.pytorch_to_caffe2(
            model,
            self.dummy_model_input,
            self.input_names,
            self.output_names,
            export_path,
            export_onnx_path,
        )

        # prepend operators
        init_net, predict_net, final_input_names = self.prepend_operators(
            init_net, predict_net, self.input_names
        )
        init_net = core.Net(init_net)
        predict_net = core.Net(predict_net)

        # postprocess input
        mobile_onnx.create_context(init_net)
        net_outputs, final_out_names = self.postprocess_output(
            init_net, predict_net, workspace, self.output_names, model
        )
        for output in net_outputs:
            predict_net.AddExternalOutput(output)

        # convert nets to proto
        init_net = init_net.Proto()
        predict_net = predict_net.Proto()

        # save proto files
        save_caffe2_pb_net(init_net_path, init_net)
        save_caffe2_pb_net(predict_net_path, predict_net)


EXPORTER_MAP = {
    ExporterType.PREDICTOR: ModelExporter,
    ExporterType.INIT_PREDICT: InitPredictNetExporter,
}


def get_exporter(name):
    exporter = EXPORTER_MAP.get(name, None)
    if not exporter:
        raise NotImplementedError
    return exporter
