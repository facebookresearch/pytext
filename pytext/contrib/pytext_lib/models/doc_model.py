#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.field_config import EmbedInitStrategy
from pytext.config.module_config import PoolingType
from pytext.contrib.pytext_lib.models.classification_heads import (
    BinaryClassificationHead,
)
from pytext.contrib.pytext_lib.transforms import build_vocab
from pytext.data.utils import Vocabulary
from pytext.models.embeddings import WordEmbedding
from pytext.utils.embeddings import PretrainedEmbedding

from .mlp_decoder import MLPDecoder


class DocNNEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_num: int = 100,
        kernel_sizes: Optional[Sequence[int]] = None,
        pooling_type: PoolingType = PoolingType.MAX,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.max_kernel = max(kernel_sizes)
        self.convs: nn.ModuleList = nn.ModuleList(
            [nn.Conv1d(embed_dim, kernel_num, k, padding=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.pooling_type = pooling_type
        self.out_dim = len(kernel_sizes) * kernel_num

    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        # embedded_tokens of size (N,W,D)
        x = embedded_tokens
        # nn.Conv1d expects a tensor of dim (batch_size x embed_dim x seq_len)
        x = x.transpose(1, 2)
        # hard code max pooling to make it torchscriptable
        x = [torch.max(F.relu(conv(x)), dim=2)[0] for conv in self.convs]
        # x = [self.convs[0](x)]
        x = self.dropout(torch.cat(x, 1))  # (N,len(Ks)*Co)
        return x


class WordEmbedding(nn.Module):
    def __init__(
        self,
        pretrained_embeddings_path: str,
        vocab: Vocabulary,
        embedding_dim: int,
        mlp_layer_dims: Optional[Sequence[int]] = None,
        lowercase_tokens: bool = False,
        skip_header: bool = True,
        delimiter: str = " ",
    ) -> None:
        super().__init__()
        pretrained_embedding = PretrainedEmbedding(
            pretrained_embeddings_path,
            lowercase_tokens=lowercase_tokens,
            skip_header=skip_header,
            delimiter=delimiter,
        )
        embeddings_weight = pretrained_embedding.initialize_embeddings_weights(
            vocab.idx,  # tensorizer.vocab.idx,
            vocab.unk_token,  # tensorizer.vocab.unk_token,
            embedding_dim,
            EmbedInitStrategy.RANDOM,
        )
        num_embeddings = len(vocab.idx)

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            _weight=embeddings_weight,
            padding_idx=vocab.get_pad_index(),
        )

        # Initialize unk embedding with zeros
        # to guard the model against randomized decisions based on unknown words
        unk_token_idx = vocab.get_unk_index()
        if unk_token_idx >= 0:
            self.embedding.weight.data[unk_token_idx].fill_(0.0)

        # Create MLP layers
        if mlp_layer_dims is None:
            mlp_layer_dims = []

        self.mlp = nn.Sequential(
            *(
                nn.Sequential(nn.Linear(m, n), nn.ReLU())
                for m, n in zip([embedding_dim] + list(mlp_layer_dims), mlp_layer_dims)
            )
        )
        self.output_dim = mlp_layer_dims[-1] if mlp_layer_dims else embedding_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_emb = self.embedding(tokens)
        return self.mlp(token_emb)

    def get_output_dim(self):
        return self.output_dim


class DocModel(nn.Module):
    def __init__(
        self,
        # word embedding object
        word_embedding: WordEmbedding,
        # DocNN config
        kernel_num: int = 100,
        kernel_sizes: Optional[Sequence[int]] = None,
        pooling_type: PoolingType = PoolingType.MAX,
        dropout: float = 0.4,
        # decoder config
        dense_dim=0,
        decoder_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.word_embedding = word_embedding
        self.encoder = DocNNEncoder(
            embed_dim=self.word_embedding.get_output_dim(),
            kernel_num=kernel_num,
            kernel_sizes=kernel_sizes,
            pooling_type=pooling_type,
            dropout=dropout,
        )
        self.decoder = MLPDecoder(
            in_dim=self.encoder.out_dim + dense_dim,
            out_dim=2,
            bias=True,
            hidden_dims=decoder_hidden_dims,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = inputs["token_ids"]
        denses = inputs["dense"] if "dense" in inputs else None
        word_embedding_output = self.word_embedding(tokens)
        encoder_output = self.encoder(word_embedding_output)
        return self.decoder(encoder_output, denses)


class DocModelForBinaryDocClassification(nn.Module):
    def __init__(
        self,
        word_embedding: WordEmbedding,
        # DocNN config
        kernel_num: int = 100,
        kernel_sizes: Optional[Sequence[int]] = None,
        pooling_type: PoolingType = PoolingType.MAX,
        dropout: float = 0.4,
        # decoder config
        dense_dim: int = 0,
        decoder_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.doc_model = DocModel(
            word_embedding=word_embedding,
            kernel_num=kernel_num,
            kernel_sizes=kernel_sizes,
            pooling_type=pooling_type,
            dropout=dropout,
            dense_dim=dense_dim,
            decoder_hidden_dims=decoder_hidden_dims,
        )
        self.head = BinaryClassificationHead()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        logits = self.doc_model(inputs)
        return self.head(logits)

    def get_loss(self, inputs: Dict[str, torch.Tensor], targets) -> torch.Tensor:
        logits = self.doc_model(inputs)
        return self.head.get_loss(logits, targets)


def build_model(
    model_name: str,
    word_embedding_file: str,
    embedding_dim: int,
    mlp_layer_dims: Sequence[int],
    lowercase_tokens: bool,
    skip_header: bool,
    kernel_num: int,
    kernel_sizes: Sequence[int],
    dropout: int,
    dense_dim: int,
    decoder_hidden_dims: Sequence[int],
):
    """build a custom doc model
    """
    vocab = build_vocab(word_embedding_file)
    word_embedding = WordEmbedding(
        pretrained_embeddings_path=word_embedding_file,
        vocab=vocab,
        embedding_dim=embedding_dim,
        mlp_layer_dims=mlp_layer_dims,
        lowercase_tokens=lowercase_tokens,
        skip_header=skip_header,
    )
    if model_name == DocModelForBinaryDocClassification.__name__:
        doc_model = DocModelForBinaryDocClassification(
            word_embedding=word_embedding,
            # DocNN config
            kernel_num=kernel_num,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            # decoder config
            dense_dim=dense_dim,
            decoder_hidden_dims=decoder_hidden_dims,
        )
    else:
        raise RuntimeError(f"unknown model_name: {model_name}")
    return doc_model


def doc_model_with_spm_embedding(
    model_name: str,
    word_embedding_file: str,
    embedding_dim: int = 1024,
    mlp_layer_dims: Sequence[int] = (150,),
    lowercase_tokens: bool = False,
    skip_header: bool = False,
    kernel_num: int = 100,
    kernel_sizes: Sequence[int] = (3, 4, 5),
    dropout: int = 0.25,
    dense_dim: int = 0,
    decoder_hidden_dims: Sequence[int] = (128,),
):
    """a shortcut to build a doc model with the best defaults for spm embedding
    """
    return build_model(
        model_name=DocModelForBinaryDocClassification.__name__,
        word_embedding_file=word_embedding_file,
        embedding_dim=embedding_dim,
        mlp_layer_dims=mlp_layer_dims,
        lowercase_tokens=lowercase_tokens,
        skip_header=skip_header,
        kernel_num=kernel_num,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        dense_dim=dense_dim,
        decoder_hidden_dims=decoder_hidden_dims,
    )


def doc_model_with_xlu_embedding(
    word_embedding_file: str,
    embedding_dim: int = 300,
    mlp_layer_dims: Sequence[int] = (256,),
    lowercase_tokens: bool = False,
    skip_header: bool = True,
    kernel_num: int = 100,
    kernel_sizes: Sequence[int] = (3, 4, 5),
    dropout: int = 0.25,
    dense_dim: int = 0,
    decoder_hidden_dims: Sequence[int] = (128,),
):
    """a shortcut to build a doc model with the best defaults for xlu embedding
    """
    return build_model(
        model_name=DocModelForBinaryDocClassification.__name__,
        word_embedding_file=word_embedding_file,
        embedding_dim=embedding_dim,
        mlp_layer_dims=mlp_layer_dims,
        lowercase_tokens=lowercase_tokens,
        skip_header=skip_header,
        kernel_num=kernel_num,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        dense_dim=dense_dim,
        decoder_hidden_dims=decoder_hidden_dims,
    )
