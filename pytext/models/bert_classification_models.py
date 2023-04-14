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
from pytext.models.representations.huggingface_electra_sentence_encoder import (  # noqa
    HuggingFaceElectraSentenceEncoder,
)
from pytext.models.representations.representation_base import RepresentationBase
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)
from pytext.torchscript.module import (
    ScriptPyTextEmbeddingModuleIndex,
    ScriptPyTextEmbeddingModuleWithDenseIndex,
)
from pytext.torchscript.utils import ScriptBatchInput, squeeze_1d, squeeze_2d
from pytext.utils.label import get_label_weights
from pytext.utils.usage import log_class_usage


class _EncoderBaseModel(BaseModel):
    """
    Classification model following the pattern of tensorizer + encoder.
    """

    SUPPORT_FP16_OPTIMIZER = True

    class Config(BaseModel.Config):
        class EncoderModelInput(BaseModel.Config.ModelInput):
            tokens: Tensorizer.Config = Tensorizer.Config()
            dense: Optional[FloatListTensorizer.Config] = None
            labels: LabelTensorizer.Config = LabelTensorizer.Config()
            # for metric reporter
            num_tokens: NtokensTensorizer.Config = NtokensTensorizer.Config(
                names=["tokens"], indexes=[2]
            )

        inputs: EncoderModelInput = EncoderModelInput()
        encoder: RepresentationBase.Config
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
        if getattr(config, "use_selfie", False):
            # No MLP fusion in SELFIE
            dense_dim = 0
        else:
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

        additional_kwargs = {}
        if hasattr(config, "r3f_options"):
            additional_kwargs["r3f_options"] = config.r3f_options

        output_layer = output_layer_cls(list(labels), loss)
        return cls(encoder, decoder, output_layer, **additional_kwargs)

    def __init__(self, encoder, decoder, output_layer, stage=Stage.TRAIN) -> None:
        super().__init__(stage=stage)
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = output_layer
        self.stage = stage
        self.module_list = [encoder, decoder]
        log_class_usage(__class__)


class NewBertModel(_EncoderBaseModel):
    """BERT single sentence classification."""

    class Config(_EncoderBaseModel.Config):
        class BertModelInput(_EncoderBaseModel.Config.EncoderModelInput):
            tokens: BERTTensorizer.Config = BERTTensorizer.Config(max_seq_len=128)

        inputs: BertModelInput = BertModelInput()
        encoder: TransformerSentenceEncoderBase.Config = (
            HuggingFaceBertSentenceEncoder.Config()
        )


class _EncoderPairwiseModel(BasePairwiseModel):
    """
    Pairwise classification model following the pattern of two tensorizers + two
    (usually shared) encoders. Also supports exporting a single tensorizer + encoder
    to produce a sentence embedding.
    """

    class Config(BasePairwiseModel.Config):
        class EncoderPairwiseModelInput(ModelInputBase):
            tokens1: Tensorizer.Config = Tensorizer.Config()
            tokens2: Tensorizer.Config = Tensorizer.Config()
            labels: LabelTensorizer.Config = LabelTensorizer.Config()
            # for metric reporter
            num_tokens: NtokensTensorizer.Config = NtokensTensorizer.Config(
                names=["tokens1", "tokens2"], indexes=[2, 2]
            )

        inputs: EncoderPairwiseModelInput = EncoderPairwiseModelInput()
        encoder: RepresentationBase.Config
        # Decoder is a fully connected layer that expects concatenated encodings.
        # So, if decoder is provided we will concatenate the encodings from the
        # encoders and then pass to the decoder.
        decoder: Optional[MLPDecoder.Config] = MLPDecoder.Config()
        shared_encoder: bool = True

    def __init__(
        self,
        encoder1,
        encoder2,
        decoder,
        output_layer,
        encode_relations,
        shared_encoder,
    ) -> None:
        super().__init__(decoder, output_layer, encode_relations)
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.shared_encoder = shared_encoder
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
            config.output_layer,
            # in subclass of this model, the labels tensorizer does not have a vocab
            labels=getattr(tensorizers["labels"], "vocab", None),
        )
        return cls(
            encoder1,
            encoder2,
            decoder,
            output_layer,
            config.encode_relations,
            config.shared_encoder,
        )

    def arrange_model_inputs(self, tensor_dict):
        return tensor_dict["tokens1"], tensor_dict["tokens2"]

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def _encoder_forwards(self, input_tuple1, input_tuple2):
        return self.encoder1(input_tuple1)[0], self.encoder2(input_tuple2)[0]

    def forward(
        self,
        input_tuple1: Tuple[torch.Tensor, ...],
        input_tuple2: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        encodings = self._encoder_forwards(input_tuple1, input_tuple2)
        if self.encode_relations:
            encodings = self._encode_relations(encodings)
        return self.decoder(torch.cat(encodings, -1)) if self.decoder else encodings

    def save_modules(self, base_path: str = "", suffix: str = ""):
        modules = {}
        if not self.shared_encoder:
            # need to save both encoders
            modules = {"encoder1": self.encoder1, "encoder2": self.encoder2}
        self._save_modules(modules, base_path, suffix)

    def torchscriptify(self, tensorizers, traced_model, trace_both_encoders):
        if trace_both_encoders:

            class ScriptModel(torch.jit.ScriptModule):
                def __init__(self, model, tensorizer1, tensorizer2):
                    super().__init__()
                    self.model = model
                    self.tensorizer1 = tensorizer1
                    self.tensorizer2 = tensorizer2

                @torch.jit.script_method
                def forward(
                    self,
                    # first input
                    texts1: Optional[List[str]] = None,
                    tokens1: Optional[List[List[str]]] = None,
                    # second input
                    texts2: Optional[List[str]] = None,
                    tokens2: Optional[List[List[str]]] = None,
                ):
                    inputs1: ScriptBatchInput = ScriptBatchInput(
                        texts=squeeze_1d(texts1),
                        tokens=squeeze_2d(tokens1),
                        languages=None,
                    )
                    inputs2: ScriptBatchInput = ScriptBatchInput(
                        texts=squeeze_1d(texts2),
                        tokens=squeeze_2d(tokens2),
                        languages=None,
                    )
                    input_tensors1 = self.tensorizer1(inputs1)
                    input_tensors2 = self.tensorizer2(inputs2)
                    return self.model(input_tensors1, input_tensors2)

            tensorizer1 = tensorizers["tokens1"].torchscriptify()
            tensorizer2 = tensorizers["tokens2"].torchscriptify()
            return ScriptModel(traced_model, tensorizer1, tensorizer2)
        else:
            # optionally trace only one encoder
            script_tensorizer = tensorizers["tokens1"].torchscriptify()
            if "dense" in tensorizers:
                return ScriptPyTextEmbeddingModuleWithDenseIndex(
                    model=traced_model,
                    tensorizer=script_tensorizer,
                    normalizer=tensorizers["dense"].normalizer,
                    index=0,
                )
            else:
                return ScriptPyTextEmbeddingModuleIndex(
                    model=traced_model, tensorizer=script_tensorizer, index=0
                )


class BertPairwiseModel(_EncoderPairwiseModel):
    """
    Bert Pairwise classification model

    The model takes two sets of tokens (left and right) and calculates their
    representations separately using shared BERT encoder. The final prediction can
    be the cosine similarity of the embeddings, or if encoder_relations is specified the
    concatenation of the embeddings, their absolute difference, and elementwise product.
    """

    class Config(_EncoderPairwiseModel.Config):
        class BertPairwiseModelInput(
            _EncoderPairwiseModel.Config.EncoderPairwiseModelInput
        ):
            tokens1: BERTTensorizerBase.Config = BERTTensorizer.Config(
                columns=["text1"], max_seq_len=128
            )
            tokens2: BERTTensorizerBase.Config = BERTTensorizer.Config(
                columns=["text2"], max_seq_len=128
            )

        inputs: BertPairwiseModelInput = BertPairwiseModelInput()
        encoder: TransformerSentenceEncoderBase.Config = (
            HuggingFaceBertSentenceEncoder.Config()
        )
