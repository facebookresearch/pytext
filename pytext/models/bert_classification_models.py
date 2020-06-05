#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.config.component import create_loss
from pytext.data.bert_tensorizer import BERTTensorizer, BERTTensorizerBase
from pytext.data.dense_retrieval_tensorizer import (  # noqa
    BERTContextTensorizerForDenseRetrieval,
    PositiveLabelTensorizerForDenseRetrieval,
)
from pytext.data.tensorizers import (
    FloatListTensorizer,
    LabelTensorizer,
    NtokensTensorizer,
    Tensorizer,
)
from pytext.loss import BinaryCrossEntropyLoss, MultiLabelSoftMarginLoss
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import BaseModel, ModelInputBase
from pytext.models.module import create_module
from pytext.models.output_layers import ClassificationOutputLayer
from pytext.models.output_layers.doc_classification_output_layer import (
    BinaryClassificationOutputLayer,
    MulticlassOutputLayer,
    MultiLabelOutputLayer,
)
from pytext.models.pair_classification_model import BasePairwiseModel
from pytext.models.representations.huggingface_bert_sentence_encoder import (
    HuggingFaceBertSentenceEncoder,
)
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)
from pytext.utils.label import get_label_weights
from pytext.utils.usage import log_class_usage


class NewBertModel(BaseModel):
    """BERT single sentence classification."""

    SUPPORT_FP16_OPTIMIZER = True

    class Config(BaseModel.Config):
        class BertModelInput(BaseModel.Config.ModelInput):
            tokens: BERTTensorizer.Config = BERTTensorizer.Config(max_seq_len=128)
            dense: Optional[FloatListTensorizer.Config] = None
            labels: LabelTensorizer.Config = LabelTensorizer.Config()
            # for metric reporter
            num_tokens: NtokensTensorizer.Config = NtokensTensorizer.Config(
                names=["tokens"], indexes=[2]
            )

        inputs: BertModelInput = BertModelInput()
        encoder: TransformerSentenceEncoderBase.Config = (
            HuggingFaceBertSentenceEncoder.Config()
        )
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )

    def arrange_model_inputs(self, tensor_dict):
        model_inputs = (tensor_dict["tokens"],)
        if "dense" in tensor_dict:
            model_inputs += (tensor_dict["dense"],)
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
        dense_dim = tensorizers["dense"].dim if "dense" in tensorizers else 0
        decoder = create_module(
            config.decoder,
            in_dim=encoder.representation_dim + dense_dim,
            out_dim=len(labels),
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


class BertPairwiseModel(BasePairwiseModel):
    """Bert Pairwise classification model

    The model takes two sets of tokens (left and right), calculates their
    representations separately using shared BERT encoder and passes them to
    the decoder along with their absolute difference and elementwise product,
    all concatenated. Used for e.g. natural language inference.
    """

    class Config(BasePairwiseModel.Config):
        class ModelInput(ModelInputBase):
            tokens1: BERTTensorizerBase.Config = BERTTensorizer.Config(
                columns=["text1"], max_seq_len=128
            )
            tokens2: BERTTensorizerBase.Config = BERTTensorizer.Config(
                columns=["text2"], max_seq_len=128
            )
            labels: LabelTensorizer.Config = LabelTensorizer.Config()
            # for metric reporter
            num_tokens: NtokensTensorizer.Config = NtokensTensorizer.Config(
                names=["tokens1", "tokens2"], indexes=[2, 2]
            )

        inputs: ModelInput = ModelInput()
        encoder: TransformerSentenceEncoderBase.Config = (
            HuggingFaceBertSentenceEncoder.Config()
        )
        # Decoder is a fully connected layer that expects concatenated encodings.
        # So, if decoder is provided we will concatenate the encodings from the
        # encoders and then pass to the decoder.
        decoder: Optional[MLPDecoder.Config] = MLPDecoder.Config()
        shared_encoder: bool = True

    def __init__(
        self, encoder1, encoder2, decoder, output_layer, encode_relations
    ) -> None:
        super().__init__(decoder, output_layer, encode_relations)
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoders = [encoder1, encoder2]
        log_class_usage(__class__)

    @classmethod
    def _create_encoder(
        cls, config: Config, tensorizers: Dict[str, Tensorizer]
    ) -> nn.ModuleList:
        encoder1 = create_module(
            config.encoder,
            output_encoded_layers=False,
            padding_idx=tensorizers["tokens1"].vocab.get_pad_index(),
            vocab_size=len(tensorizers["tokens1"].vocab),
        )
        if config.shared_encoder:
            encoder2 = encoder1
        else:
            encoder2 = create_module(
                config.encoder,
                output_encoded_layers=False,
                padding_idx=tensorizers["tokens2"].vocab.get_pad_index(),
                vocab_size=len(tensorizers["tokens2"].vocab),
            )
        return encoder1, encoder2

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        encoder1, encoder2 = cls._create_encoder(config, tensorizers)
        decoder = cls._create_decoder(config, [encoder1, encoder2], tensorizers)
        output_layer = create_module(
            config.output_layer, labels=tensorizers["labels"].vocab
        )
        return cls(encoder1, encoder2, decoder, output_layer, config.encode_relations)

    def arrange_model_inputs(self, tensor_dict):
        return tensor_dict["tokens1"], tensor_dict["tokens2"]

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def forward(
        self,
        input_tuple1: Tuple[torch.Tensor, ...],
        input_tuple2: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        encodings = [self.encoder1(input_tuple1)[0], self.encoder2(input_tuple2)[0]]
        if self.encode_relations:
            encodings = self._encode_relations(encodings)
        return self.decoder(torch.cat(encodings, -1)) if self.decoder else encodings

    def save_modules(self, base_path: str = "", suffix: str = ""):
        self._save_modules(self.encoders, base_path, suffix)
