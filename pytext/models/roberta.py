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
from pytext.torchscript.module import get_script_module_cls
from pytext.utils.file_io import PathManager
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
            load_path=(
                "manifold://pytext_training/tree/static/models/roberta_public.pt1"
            )
        )

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        self.encoder = create_module(config.pretrained_encoder)
        self.representation_dim = self.encoder.encoder.token_embedding.weight.size(-1)

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

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.encoder.transformer.token_embedding


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
        script_tensorizer = tensorizers["tokens"].torchscriptify()
        script_module_cls = get_script_module_cls(
            script_tensorizer.tokenizer.input_type()
        )

        return script_module_cls(
            model=traced_model,
            output_layer=self.output_layer.torchscript_predictions(),
            tensorizer=script_tensorizer,
        )
