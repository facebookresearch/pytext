#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional, Tuple

import torch
from pytext import resources
from pytext.common.constants import Stage
from pytext.config import ConfigBase
from pytext.data.roberta_tensorizer import (
    RoBERTaTensorizer,
    RoBERTaTokenLevelTensorizer,
)
from pytext.data.tensorizers import (
    FloatListTensorizer,
    LabelTensorizer,
    NumericLabelTensorizer,
    Tensorizer,
)
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.bert_regression_model import NewBertRegressionModel
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import BaseModel
from pytext.models.module import Module, create_module
from pytext.models.output_layers import WordTaggingOutputLayer
from pytext.models.representations.transformer import (
    MultiheadSelfAttention,
    SentenceEncoder,
    Transformer,
    TransformerLayer,
)
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)
from pytext.torchscript.module import ScriptPyTextModule, ScriptPyTextModuleWithDense
from pytext.utils.file_io import PathManager
from pytext.utils.usage import log_class_usage
from torch.serialization import default_restore_location


def init_params(module):
    """Initialize the RoBERTa weights for pre-training from scratch."""

    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class RoBERTaEncoderBase(TransformerSentenceEncoderBase):
    __EXPANSIBLE__ = True

    class Config(TransformerSentenceEncoderBase.Config):
        pass

    def _encoder(self, inputs):
        # NewBertModel expects the output as a tuple and grabs the first element
        tokens, _, _, _ = inputs
        full_representation = self.encoder(tokens)
        sentence_rep = full_representation[:, 0, :]
        return [full_representation], sentence_rep


class RoBERTaEncoderJit(RoBERTaEncoderBase):
    """A TorchScript RoBERTa implementation"""

    class Config(RoBERTaEncoderBase.Config):
        pretrained_encoder: Module.Config = Module.Config(
            load_path=resources.roberta.PUBLIC
        )

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        config.pretrained_encoder.load_path = (
            resources.roberta.RESOURCE_MAP[config.pretrained_encoder.load_path]
            if config.pretrained_encoder.load_path in resources.roberta.RESOURCE_MAP
            else config.pretrained_encoder.load_path
        )
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        self.encoder = create_module(config.pretrained_encoder)
        self.representation_dim = self.encoder.encoder.token_embedding.weight.size(-1)
        log_class_usage(__class__)

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.encoder.encoder.token_embedding


class RoBERTaEncoder(RoBERTaEncoderBase):
    """A PyTorch RoBERTa implementation"""

    class Config(RoBERTaEncoderBase.Config):
        embedding_dim: int = 768
        vocab_size: int = 50265
        num_encoder_layers: int = 12
        num_attention_heads: int = 12
        model_path: str = (
            "manifold://pytext_training/tree/static/models/roberta_base_torch.pt"
        )
        # Loading the state dict of the model depends on whether the model was
        # previously finetuned in PyText or not. If it was finetuned then we
        # dont need to translate the state dict and can just load it`
        # directly.
        is_finetuned: bool = False

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)

        # map to the real model_path
        config.model_path = (
            resources.roberta.RESOURCE_MAP[config.model_path]
            if config.model_path in resources.roberta.RESOURCE_MAP
            else config.model_path
        )
        # assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        self.encoder = SentenceEncoder(
            transformer=Transformer(
                vocab_size=config.vocab_size,
                embedding_dim=config.embedding_dim,
                layers=[
                    TransformerLayer(
                        embedding_dim=config.embedding_dim,
                        attention=MultiheadSelfAttention(
                            config.embedding_dim, config.num_attention_heads
                        ),
                    )
                    for _ in range(config.num_encoder_layers)
                ],
            )
        )
        self.apply(init_params)
        if config.model_path:
            with PathManager.open(config.model_path, "rb") as f:
                roberta_state = torch.load(
                    f, map_location=lambda s, l: default_restore_location(s, "cpu")
                )
            # In case the model has previously been loaded in PyText and finetuned,
            # then we dont need to do the special state dict translation. Load
            # it directly
            if not config.is_finetuned:
                self.encoder.load_roberta_state_dict(roberta_state["model"])
            else:
                self.load_state_dict(roberta_state)

        self.representation_dim = self._embedding().weight.size(-1)
        log_class_usage(__class__)

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.encoder.transformer.token_embedding


class RoBERTa(NewBertModel):
    class Config(NewBertModel.Config):
        class InputConfig(ConfigBase):
            tokens: RoBERTaTensorizer.Config = RoBERTaTensorizer.Config()
            dense: Optional[FloatListTensorizer.Config] = None
            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: InputConfig = InputConfig()
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoderJit.Config()

    def torchscriptify(self, tensorizers, traced_model):
        """Using the traced model, create a ScriptModule which has a nicer API that
        includes generating tensors from simple data types, and returns classified
        values according to the output layer (eg. as a dict mapping class name to score)
        """
        script_tensorizer = tensorizers["tokens"].torchscriptify()
        if "dense" in tensorizers:
            return ScriptPyTextModuleWithDense(
                model=traced_model,
                output_layer=self.output_layer.torchscript_predictions(),
                tensorizer=script_tensorizer,
                normalizer=tensorizers["dense"].normalizer,
            )
        else:
            return ScriptPyTextModule(
                model=traced_model,
                output_layer=self.output_layer.torchscript_predictions(),
                tensorizer=script_tensorizer,
            )


class RoBERTaRegression(NewBertRegressionModel):
    class Config(NewBertRegressionModel.Config):
        class RegressionModelInput(ConfigBase):
            tokens: RoBERTaTensorizer.Config = RoBERTaTensorizer.Config()
            labels: NumericLabelTensorizer.Config = NumericLabelTensorizer.Config()

        inputs: RegressionModelInput = RegressionModelInput()
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoderJit.Config()

    def torchscriptify(self, tensorizers, traced_model):
        """Using the traced model, create a ScriptModule which has a nicer API that
        includes generating tensors from simple data types, and returns classified
        values according to the output layer (eg. as a dict mapping class name to score)
        """
        script_tensorizer = tensorizers["tokens"].torchscriptify()
        return ScriptPyTextModule(
            model=traced_model,
            output_layer=self.output_layer.torchscript_predictions(),
            tensorizer=script_tensorizer,
        )


class RoBERTaWordTaggingModel(BaseModel):
    """
    Single Sentence Token-level Classification Model using XLM.
    """

    class Config(BaseModel.Config):
        class WordTaggingInputConfig(ConfigBase):
            tokens: RoBERTaTokenLevelTensorizer.Config = (
                RoBERTaTokenLevelTensorizer.Config()
            )

        inputs: WordTaggingInputConfig = WordTaggingInputConfig()
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoderJit.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: WordTaggingOutputLayer.Config = WordTaggingOutputLayer.Config()

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        label_vocab = tensorizers["tokens"].labels_vocab
        vocab = tensorizers["tokens"].vocab

        encoder = create_module(
            config.encoder,
            output_encoded_layers=True,
            padding_idx=vocab.get_pad_index(),
            vocab_size=vocab.__len__(),
        )
        decoder = create_module(
            config.decoder, in_dim=encoder.representation_dim, out_dim=len(label_vocab)
        )
        output_layer = create_module(config.output_layer, labels=label_vocab)
        return cls(encoder, decoder, output_layer)

    def __init__(self, encoder, decoder, output_layer, stage=Stage.TRAIN) -> None:
        super().__init__(stage=stage)
        self.encoder = encoder
        self.decoder = decoder
        self.module_list = [encoder, decoder]
        self.output_layer = output_layer
        self.stage = stage
        log_class_usage(__class__)

    def arrange_model_inputs(self, tensor_dict):
        tokens, pad_mask, segment_labels, positions, _ = tensor_dict["tokens"]
        model_inputs = (tokens, pad_mask, segment_labels, positions)
        return (model_inputs,)

    def arrange_targets(self, tensor_dict):
        _, _, _, _, labels = tensor_dict["tokens"]
        return labels

    def forward(self, encoder_inputs: Tuple[torch.Tensor, ...], *args) -> torch.Tensor:
        # The encoder outputs a list of representations for each token where
        # every element of the list corresponds to a layer in the transformer.
        # We extract and pass the representations associated with the last layer
        # of the transformer.
        representation = self.encoder(encoder_inputs)[0][-1]
        return self.decoder(representation, *args)
