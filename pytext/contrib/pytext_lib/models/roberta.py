#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from pytext.config.module_config import Activation
from pytext.data.utils import Vocabulary
from pytext.loss import BinaryCrossEntropyLoss
from pytext.models.output_layers.doc_classification_output_layer import (
    BinaryClassificationOutputLayer,
)
from pytext.models.representations.transformer import (
    MultiheadSelfAttention,
    Transformer,
    TransformerLayer,
)
from pytext.models.representations.transformer.sentence_encoder import (
    translate_roberta_state_dict,
)
from pytext.optimizer import get_activation
from pytext.utils.file_io import PathManager
from torch.serialization import default_restore_location


class RoBERTaEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_attention_heads: int,
        num_encoder_layers: int,
        output_dropout: float,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.transformer = Transformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            layers=[
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    attention=MultiheadSelfAttention(
                        embedding_dim, num_attention_heads
                    ),
                )
                for _ in range(num_encoder_layers)
            ],
        )
        self.output_dropout = nn.Dropout(output_dropout)

        self.apply(init_params)
        if model_path:
            with PathManager.open(model_path, "rb") as f:
                roberta_state = torch.load(
                    f, map_location=lambda s, l: default_restore_location(s, "cpu")
                )
                if "model" in roberta_state:
                    roberta_state = translate_roberta_state_dict(roberta_state["model"])
                self.load_state_dict(roberta_state)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        all_layers = self.transformer(tokens)  # lists of T x B x C
        last_layer = all_layers[-1].transpose(0, 1)
        sentence_rep = last_layer[:, 0, :]
        return self.output_dropout(sentence_rep)


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


class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        hidden_dims: List[int] = None,
        activation: Activation = Activation.RELU,
    ) -> None:
        super().__init__()
        layers = []
        for dim in hidden_dims or []:
            layers.append(nn.Linear(in_dim, dim, bias))
            layers.append(get_activation(activation))
            in_dim = dim
        layers.append(nn.Linear(in_dim, out_dim, bias))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, representation: torch.Tensor, dense: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if dense:
            representation = torch.cat([representation, dense], 1)
        return self.mlp(representation)


class RobertaModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = inputs["tokens"]
        dense = inputs.get("dense", None)

        representation = self.encoder(tokens)
        return self.decoder(representation, dense=dense)


class BinaryClassificationHead(nn.Module):  # similar to ClassificationOutputLayer
    def __init__(self, label_vocab, label_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        loss = BinaryCrossEntropyLoss.from_config(BinaryCrossEntropyLoss.Config())
        self.output_layer = BinaryClassificationOutputLayer(label_vocab, loss)

    def forward(self, logits):
        return self.output_layer.get_pred(logits)

    def get_loss(self, logits, targets):
        return self.output_layer.get_loss(logits, targets)


class RobertaModelForBinaryDocClassification(RobertaModel):
    def __init__(self, label_vocab, encoder: nn.Module, decoder: nn.Module):
        super().__init__(encoder, decoder)
        self.head = BinaryClassificationHead(label_vocab)

    def get_pred(self, logits: torch.Tensor) -> torch.Tensor:
        return self.head(logits)

    def get_loss(self, logits, targets):
        return self.head.get_loss(logits, targets)


def roberta_base_binary_doc_classifier(pretrained=True):
    model_path = (
        "manifold://pytext_training/tree/static/models/roberta_base_torch.pt"
        if pretrained
        else None
    )
    dense_dim = 0
    embedding_dim = 768

    encoder = RoBERTaEncoder(
        vocab_size=50265,
        embedding_dim=embedding_dim,
        num_attention_heads=12,
        num_encoder_layers=12,
        output_dropout=0.4,
        model_path=model_path,
    )
    decoder = MLPDecoder(in_dim=embedding_dim + dense_dim, out_dim=2, bias=True)

    label_vocab = Vocabulary(["1", "0"])
    model = RobertaModelForBinaryDocClassification(
        label_vocab=label_vocab, encoder=encoder, decoder=decoder
    )
    return model


def xlm_roberta_base_binary_doc_classifier(pretrained=True):
    model_path = (
        "manifold://nlp_technologies/tree/xlm/models/xlm_r/checkpoint_base_1500k.pt"
        if pretrained
        else None
    )

    dense_dim = 0
    embedding_dim = 768

    encoder = RoBERTaEncoder(
        vocab_size=250002,
        embedding_dim=embedding_dim,
        num_attention_heads=12,
        num_encoder_layers=12,
        output_dropout=0.4,
        model_path=model_path,
    )
    decoder = MLPDecoder(in_dim=embedding_dim + dense_dim, out_dim=2, bias=True)

    label_vocab = Vocabulary(["1", "0"])
    model = RobertaModelForBinaryDocClassification(
        label_vocab=label_vocab, encoder=encoder, decoder=decoder
    )
    return model


def xlm_roberta_dummy_binary_doc_classifier(pretrained=False):
    model_path = (
        "manifold://nlp_technologies/tree/pytext/public/models/xlmr/xlmr_dummy.pt"
        if pretrained
        else None
    )

    dense_dim = 0
    embedding_dim = 10

    encoder = RoBERTaEncoder(
        vocab_size=250002,
        embedding_dim=embedding_dim,
        num_attention_heads=1,
        num_encoder_layers=1,
        output_dropout=0.4,
        model_path=model_path,
    )
    decoder = MLPDecoder(in_dim=embedding_dim + dense_dim, out_dim=2, bias=True)

    label_vocab = Vocabulary(["1", "0"])
    model = RobertaModelForBinaryDocClassification(
        label_vocab=label_vocab, encoder=encoder, decoder=decoder
    )
    return model
