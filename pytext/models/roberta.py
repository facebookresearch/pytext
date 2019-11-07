#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch
from pytext.config import ConfigBase
from pytext.data.roberta_tensorizer import RoBERTaTensorizer
from pytext.data.tensorizers import LabelTensorizer
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.module import Module, create_module
from pytext.models.representations.transformer import (
    MultiheadSelfAttention,
    SentenceEncoder,
    Transformer,
    TransformerLayer,
)
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)
from pytext.torchscript.tensorizer import ScriptRoBERTaTensorizer
from torch.serialization import default_restore_location


class RoBERTaEncoderBase(TransformerSentenceEncoderBase):
    __EXPANSIBLE__ = True

    class Config(TransformerSentenceEncoderBase.Config):
        pass

    def _encoder(self, inputs):
        # NewBertModel expects the output as a tuple and grabs the first element
        tokens, _, _ = inputs
        full_representation = self.encoder(tokens)
        sentence_rep = full_representation[:, 0, :]
        return [full_representation], sentence_rep


class RoBERTaEncoderJit(RoBERTaEncoderBase):
    """A TorchScript RoBERTa implementation"""

    class Config(RoBERTaEncoderBase.Config):
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


class RoBERTaEncoder(RoBERTaEncoderBase):
    """A PyTorch RoBERTa implementation"""

    class Config(RoBERTaEncoderBase.Config):
        num_encoder_layers: int = 12
        num_attention_heads: int = 12
        model_path: str = (
            "manifold://pytext_training/tree/static/models/roberta_base_torch.pt"
        )

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        # assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        self.encoder = SentenceEncoder(
            transformer=Transformer(
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
        roberta_state = torch.load(
            config.model_path,
            map_location=lambda s, l: default_restore_location(s, "cpu"),
        )
        self.encoder.load_roberta_state_dict(roberta_state["model"])
        self.representation_dim = self.encoder.transformer.token_embedding.weight.size(
            -1
        )


class ScriptRoBERTa(torch.jit.ScriptModule):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        output_layer: torch.jit.ScriptModule,
        tensorizer: ScriptRoBERTaTensorizer,
    ):
        super().__init__()
        self.model = model
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
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoderJit.Config()

    def torchscriptify(self, tensorizers, traced_model):
        """Using the traced model, create a ScriptModule which has a nicer API that
        includes generating tensors from simple data types, and returns classified
        values according to the output layer (eg. as a dict mapping class name to score)
        """
        return ScriptRoBERTa(
            model=traced_model,
            output_layer=self.output_layer.torchscript_predictions(),
            tensorizer=tensorizers["tokens"].torchscriptify(),
        )
