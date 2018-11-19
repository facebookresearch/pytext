#!/usr/bin/env python3
import caffe2.caffe2.fb.predictor.predictor_exporter as pe
import numpy as np
from caffe2.python import core, dyndep, workspace
from pytext.common.constants import DatasetFieldName
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data import CommonMetadata
from pytext.exporters import ModelExporter
from pytext.models.semantic_parsers.rnng.rnng_base import RNNGParser


RNNG_INPUT_NAMES = [  # Prediction Request from www, with default values
    "tokens_vals_str:value",  # string_list
    "dict_vals_str:value",  # string_list
    "dict_weights",
    "dict_lens",
]
RNNG_INPUT_TYPES = [
    np.array(""),
    np.array(""),
    np.array([0.0], dtype=np.float32),
    np.array([0], dtype=np.int_),
]
RNNG_OUTPUT_NAMES = ["actions", "tokens", "scores", "pretty_print"]


class RNNGCppExporter(ModelExporter):
    @classmethod
    def from_config(
        cls,
        unused_config,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        metadata: CommonMetadata,
        model_config: RNNGParser.Config,
        *args,
        **kwargs
    ):
        return cls(model_config, metadata, feature_config)

    def __init__(self, model_config, metadata, feature_config) -> None:
        self.model_config = model_config
        self.metadata = metadata
        self.feature_config = feature_config

    def export_to_caffe2(self, model, export_path: str):
        workspace.GlobalInit(["caffe2", "--caffe2_log_level=2"])
        dyndep.InitOpsLibrary("//pytext/exporters/rnng_exporter:rnng_op")

        for input_name, input_type in zip(RNNG_INPUT_NAMES, RNNG_INPUT_TYPES):
            workspace.FeedBlob(input_name, input_type)

        model_state_dict = model.state_dict()
        for weight_name, weight_value in model_state_dict.items():
            workspace.FeedBlob(weight_name, weight_value.detach().numpy())

        weight_names = list(model_state_dict.keys())
        all_input_names = RNNG_INPUT_NAMES + weight_names

        predict_net = core.Net("prediction_net")
        predict_net.RNNGParser(
            all_input_names,
            RNNG_OUTPUT_NAMES,
            model_config=RNNGParser.get_cpp_model_config_list(
                self.model_config, self.feature_config
            ),
            weight_names=weight_names,
            actions_vec=self.metadata.actions_vocab.itos,
            terminals_vec=self.metadata.features[
                DatasetFieldName.TEXT_FIELD
            ].vocab.itos,
            dictfeats_vec=self.metadata.features[DatasetFieldName.DICT_FIELD].vocab.itos
            if DatasetFieldName.DICT_FIELD in self.metadata.features
            else [],  # [] because we cannot pass NULL to C++ Parser.,
        )
        workspace.RunNetOnce(predict_net)
        predictor_export_meta = pe.PredictorExportMeta(
            predict_net=predict_net,
            parameters=all_input_names,
            inputs=[],
            outputs=RNNG_OUTPUT_NAMES,
        )
        pe.save_to_db(
            db_type="log_file_db",
            db_destination=export_path,
            predictor_export_meta=predictor_export_meta,
        )


def get_model_config(pytext_config):
    task = pytext_config.task
    rnng_config = pytext_config.task.model
    model_config = [
        rnng_config.version,
        rnng_config.lstm.lstm_dim,
        rnng_config.lstm.num_layers,
        task.features.word_feat.embed_dim,
        rnng_config.max_open_NT,
        task.features.dict_feat.embed_dim if task.features.dict_feat else 0,
        rnng_config.dropout,
    ]
    return [float(mc) for mc in model_config]
