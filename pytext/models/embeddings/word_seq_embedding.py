#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

import torch
from pytext.config.field_config import EmbedInitStrategy
from pytext.data.tensorizers import Tensorizer
from pytext.models.embeddings.embedding_base import EmbeddingBase
from pytext.models.representations.bilstm import BiLSTM
from pytext.utils.embeddings import PretrainedEmbedding
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class WordSeqEmbedding(EmbeddingBase):
    """
    An embedding module represents a sequence of sentences

    Args:
        lstm_config (BiLSTM.Config): config of the lstm layer
        num_embeddings (int): Total number of words/tokens (vocabulary size).
        embedding_dim (int): Size of embedding vector.
        embeddings_weight (torch.Tensor): Pretrained weights to initialize the
            embedding table with.
        init_range (List[int]): Range of uniform distribution to initialize the
            weights with if `embeddings_weight` is None.
        unk_token_idx (int): Index of UNK token in the word vocabulary.
    """

    class Config(EmbeddingBase.Config):
        word_embed_dim: int = 100
        embedding_init_strategy: EmbedInitStrategy = EmbedInitStrategy.RANDOM
        embedding_init_range: Optional[List[float]] = None
        embeddding_init_std: Optional[float] = 0.02
        padding_idx: Optional[int] = None

        lstm: BiLSTM.Config = BiLSTM.Config()

        # [BEGIN] pretrained embedding related config
        pretrained_embeddings_path: str = ""
        #: If `pretrained_embeddings_path` and `vocab_from_pretrained_embeddings` are set,
        #: only the first `vocab_size` tokens in the file will be added to the vocab.
        vocab_size: int = 0
        lowercase_tokens: bool = True
        skip_header: bool = True
        delimiter: str = " "
        # [END] pretrained embedding related config

    @classmethod
    def from_config(
        cls,
        config: Config,
        tensorizer: Tensorizer = None,
        init_from_saved_state: Optional[bool] = False,
    ):
        """Factory method to construct an instance of WordEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (WordSeqEmbedding.Config): Configuration object specifying all the
            parameters of WordEmbedding.

        Returns:
            type: An instance of WordSeqEmbedding.
        """
        embeddings_weight = None
        # We don't need to load pretrained embeddings if we know the
        # embedding weights are going to be loaded from a snapshot.
        if config.pretrained_embeddings_path and not init_from_saved_state:
            pretrained_embedding = PretrainedEmbedding(
                config.pretrained_embeddings_path,  # doesn't support fbpkg
                lowercase_tokens=config.lowercase_tokens,
                skip_header=config.skip_header,
                delimiter=config.delimiter,
            )
            embeddings_weight = pretrained_embedding.initialize_embeddings_weights(
                tensorizer.vocab.idx,
                tensorizer.vocab.unk_token,
                config.word_embed_dim,
                config.embedding_init_strategy,
            )
        num_embeddings = len(tensorizer.vocab)
        unk_token_idx = tensorizer.vocab.get_unk_index()
        vocab = tensorizer.vocab
        vocab_pad_idx = vocab.get_pad_index(value=-1)
        if vocab_pad_idx == -1:
            vocab_pad_idx = None

        return cls(
            lstm_config=config.lstm,
            num_embeddings=num_embeddings,
            word_embed_dim=config.word_embed_dim,
            embeddings_weight=embeddings_weight,
            init_range=config.embedding_init_range,
            init_std=config.embeddding_init_std,
            unk_token_idx=unk_token_idx,
            padding_idx=config.padding_idx or vocab_pad_idx,
            vocab=vocab,
        )

    def __init__(
        self,
        lstm_config: BiLSTM.Config,
        num_embeddings: int,
        word_embed_dim: int = 300,
        embeddings_weight: Optional[torch.Tensor] = None,
        init_range: Optional[List[int]] = None,
        init_std: Optional[float] = None,
        unk_token_idx: int = 0,
        padding_idx: Optional[int] = None,
        vocab: Optional[List[str]] = None,
    ) -> None:

        lstm = BiLSTM(lstm_config, word_embed_dim)
        output_embedding_dim = lstm.representation_dim
        EmbeddingBase.__init__(self, embedding_dim=output_embedding_dim)
        self.lstm = lstm
        self.num_lstm_directions = 2 if lstm_config.bidirectional else 1

        # Create word embedding
        self.word_embedding = nn.Embedding(
            num_embeddings,
            word_embed_dim,
            _weight=embeddings_weight,
            padding_idx=padding_idx,
        )
        if embeddings_weight is None:
            if init_range:
                self.word_embedding.weight.data.uniform_(init_range[0], init_range[1])
            if init_std:
                self.word_embedding.weight.data.normal_(mean=0.0, std=init_std)

        # Initialize unk embedding with zeros
        # to guard the model against randomized decisions based on unknown words
        self.word_embedding.weight.data[unk_token_idx].fill_(0.0)

        self.vocab = vocab
        self.padding_idx = padding_idx

    def __getattr__(self, name):
        if name == "weight":
            return self.word_embedding.weight
        return super().__getattr__(name)

    def forward(self, seq_token_idx, seq_token_count):
        """
        Args:
            seq_token_idx : shape [batch_size * max_seq_len * max_token_count]
            seq_token_count : shape [batch_size * max_seq_len]
        Return:
            embedding : shape (batch_size * max_seq_len * output_dim)
        """
        # batch_size * max_seq_len * max_token_count * emb_dim
        seq_token_emb = self.word_embedding(seq_token_idx)
        # transpose to  max_seq_len * batch_size * max_token_count * emb_dim
        seq_token_emb_t = seq_token_emb.transpose(0, 1)
        # transpose to  max_seq_len * batch_size
        seq_token_count_t = seq_token_count.transpose(0, 1)

        outputs = []
        for i, token_emb in enumerate(seq_token_emb_t):
            token_count = seq_token_count_t[i]
            rep, (h_t, c_t) = self.lstm(token_emb, token_count)
            h_t_transposed = h_t.transpose(0, 1).view(
                self.lstm.config.num_layers,
                self.num_lstm_directions,
                -1,
                self.lstm.config.lstm_dim,
            )
            if self.lstm.config.bidirectional:
                # Concat the two directions of the last layer
                output = torch.cat([h_t_transposed[-1][0], h_t_transposed[-1][1]], 1)
            else:
                output = h_t_transposed[-1][0]
            # seq_len * batch_size * lstm_dim
            outputs.append(output)
        # batch_size * seq_len * lstm_dim
        return torch.stack(outputs).transpose(1, 0)

    def freeze(self):
        for param in self.word_embedding.parameters():
            param.requires_grad = False

    def visualize(self, summary_writer: SummaryWriter):
        if self.vocab:
            summary_writer.add_embedding(
                self.word_embedding.weight, metadata=self.vocab
            )
