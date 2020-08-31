#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Tuple

import torch
from pytext.common.constants import Stage
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data.dense_retrieval_tensorizer import (  # noqa
    BERTContextTensorizerForDenseRetrieval,
    PositiveLabelTensorizerForDenseRetrieval,
)
from pytext.data.roberta_tensorizer import RoBERTaTensorizer
from pytext.data.tensorizers import FloatListTensorizer, LabelTensorizer, Tensorizer
from pytext.loss import BinaryCrossEntropyLoss, MultiLabelSoftMarginLoss
from pytext.models.decoders.mlp_decoder_two_tower import MLPDecoderTwoTower
from pytext.models.model import BaseModel
from pytext.models.module import create_module
from pytext.models.output_layers import ClassificationOutputLayer
from pytext.models.output_layers.doc_classification_output_layer import (
    BinaryClassificationOutputLayer,
    MulticlassOutputLayer,
    MultiLabelOutputLayer,
)
from pytext.models.roberta import RoBERTaEncoder, RoBERTaEncoderBase
from pytext.torchscript.module import (
    ScriptPyTextEmbeddingModuleIndex,
    ScriptPyTextModule,
    ScriptPyTextTwoTowerModuleWithDense,
)
from pytext.utils.label import get_label_weights
from pytext.utils.usage import log_class_usage


class TwoTowerClassificationModel(BaseModel):

    SUPPORT_FP16_OPTIMIZER = True

    class Config(BaseModel.Config):
        class InputConfig(ConfigBase):
            tokens: RoBERTaTensorizer.Config = RoBERTaTensorizer.Config()
            right_dense: FloatListTensorizer.Config = None
            left_dense: FloatListTensorizer.Config = None

            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: InputConfig = InputConfig()
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoder.Config()
        decoder: MLPDecoderTwoTower.Config = MLPDecoderTwoTower.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )

    def trace(self, inputs):
        if self.encoder.export_encoder:
            return torch.jit.trace(self.encoder, inputs)
        else:
            return torch.jit.trace(self, inputs)

    def torchscriptify(self, tensorizers, traced_model):
        """Using the traced model, create a ScriptModule which has a nicer API that
        includes generating tensors from simple data types, and returns classified
        values according to the output layer (eg. as a dict mapping class name to score)
        """
        script_tensorizer = tensorizers["tokens"].torchscriptify()
        if self.encoder.export_encoder:
            return ScriptPyTextEmbeddingModuleIndex(
                traced_model, script_tensorizer, index=0
            )
        else:
            if "right_dense" in tensorizers and "left_dense" in tensorizers:
                return ScriptPyTextTwoTowerModuleWithDense(
                    model=traced_model,
                    output_layer=self.output_layer.torchscript_predictions(),
                    tensorizer=script_tensorizer,
                    right_normalizer=tensorizers["right_dense"].normalizer,
                    left_normalizer=tensorizers["left_dense"].normalizer,
                )
            else:
                return ScriptPyTextModule(
                    model=traced_model,
                    output_layer=self.output_layer.torchscript_predictions(),
                    tensorizer=script_tensorizer,
                )

    def arrange_model_inputs(self, tensor_dict):
        model_inputs = (
            tensor_dict["tokens"],
            tensor_dict["right_dense"],
            tensor_dict["left_dense"],
        )

        return model_inputs

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def forward(
        self, encoder_inputs: Tuple[torch.Tensor, ...], *args
    ) -> List[torch.Tensor]:
        if self.encoder.output_encoded_layers:
            # if encoded layers are returned, discard them
            representation = self.encoder(encoder_inputs)[1]
        else:
            representation = self.encoder(encoder_inputs)[0]
        return self.decoder(representation, *args)

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        labels = tensorizers["labels"].vocab
        if not labels:
            raise ValueError("Labels were not created, see preceding errors")

        vocab = tensorizers["tokens"].vocab
        encoder = create_module(
            config.encoder, padding_idx=vocab.get_pad_index(), vocab_size=len(vocab)
        )

        right_dense_dim = tensorizers["right_dense"].dim
        left_dense_dim = tensorizers["left_dense"].dim

        decoder = create_module(
            config.decoder,
            right_dim=encoder.representation_dim + right_dense_dim,
            left_dim=left_dense_dim,
            to_dim=len(labels),
        )

        label_weights = (
            get_label_weights(labels.idx, config.output_layer.label_weights)
            if config.output_layer.label_weights
            else None
        )

        loss = create_loss(config.output_layer.loss, weight=label_weights)

        if isinstance(loss, BinaryCrossEntropyLoss):
            output_layer_cls = BinaryClassificationOutputLayer
        elif isinstance(loss, MultiLabelSoftMarginLoss):
            output_layer_cls = MultiLabelOutputLayer
        else:
            output_layer_cls = MulticlassOutputLayer

        output_layer = output_layer_cls(list(labels), loss)
        return cls(encoder, decoder, output_layer)

    def __init__(self, encoder, decoder, output_layer, stage=Stage.TRAIN) -> None:
        super().__init__(stage=stage)
        self.encoder = encoder
        self.decoder = decoder
        self.module_list = [encoder, decoder]
        self.output_layer = output_layer
        self.stage = stage
        self.module_list = [encoder, decoder]
        log_class_usage(__class__)
