#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch
from pytext.config import ConfigBase
from pytext.data.bert_tensorizer import RoBERTaTensorizer
from pytext.data.tensorizers import LabelTensorizer
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.module import Module, create_module
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)
from pytext.torchscript.tensorizer import ScriptRoBERTaTensorizer


class RoBERTaEncoder(TransformerSentenceEncoderBase):
    class Config(TransformerSentenceEncoderBase.Config):
        pretrained_encoder: Module.Config = Module.Config(
            load_path=(
                "manifold://pytext_training/tree/static/models/roberta_public.pt1"
            )
        )

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        self.encoder = create_module(config.pretrained_encoder)
        self.representation_dim = self.encoder.encoder.token_embedding.weight.size(-1)

    def _encoder(self, inputs):
        # NewBertModel expects the output as a tuple and grabs the first element
        tokens, _, _ = inputs
        full_representation = self.encoder(tokens)
        sentence_rep = full_representation[:, 0, :]
        return [full_representation], sentence_rep


class ScriptRoBERTa(torch.jit.ScriptModule):
    def __init__(
        self,
        traced_model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptRoBERTaTensorizer,
    ):
        super().__init__()
        self.model = traced_model
        self.output_layer = output_layer
        self.tensorizer = tensorizer

    @torch.jit.script_method
    def forward(self, texts: List[str]):
        rows: List[List[str]] = [[text] for text in texts]
        input_tensors: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] = self.tensorizer.tensorize(rows)
        logits = self.model(input_tensors)
        return self.output_layer(logits)


class RoBERTa(NewBertModel):
    class Config(NewBertModel.Config):
        class InputConfig(ConfigBase):
            tokens: RoBERTaTensorizer.Config = RoBERTaTensorizer.Config()
            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: InputConfig = InputConfig()
        encoder: RoBERTaEncoder.Config = RoBERTaEncoder.Config()

    def torchscriptify(self, tensorizers, traced_model):
        """Using the traced model, create a ScriptModule which has a nicer API that
        includes generating tensors from simple data types, and returns classified
        values according to the output layer (eg. as a dict mapping class name to score)
        """
        output_layer = self.output_layer.torchscript_predictions()
        script_tensorizer = tensorizers["tokens"].torchscriptify()
        return ScriptRoBERTa(traced_model, output_layer, script_tensorizer)
