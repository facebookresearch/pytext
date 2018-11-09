#!/usr/bin/env python3
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.models.module import create_module
from pytext.config.module_config import SlotAttentionType

from .bilstm import BiLSTM
from .pooling import MaxPool, MeanPool, SelfAttention
from .representation_base import RepresentationBase
from .slot_attention import SlotAttention


class BiLSTMDocSlotAttention(RepresentationBase):
    """
    Multi-layer bidirectional LSTM representation with support for attention mechanisms.

    Default:
        In default mode when attention configuration is not provided, it
        behaves like a multi-layer LSTM encoder.
        It returns the output features from the last layer of the LSTM, for each t.

    Document Attention:
        When document_attention configuration is provided, it produces
        a fixed-sized document representation.

    Slot Attention:
        When slot_attention configuration is provide, it attends on output of
        each cell of LSTM module to produce a fixed sized word representation.
    """

    class Config(ConfigBase):
        dropout: float = 0.4
        lstm: BiLSTM.Config = BiLSTM.Config()
        pooling: Optional[
            Union[SelfAttention.Config, MaxPool.Config, MeanPool.Config]
        ] = None
        slot_attention: Optional[SlotAttention.Config] = None
        doc_mlp_layers: int = 0
        word_mlp_layers: int = 0

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

        # Shared representation.
        padding_value = (
            float("-inf") if isinstance(config.pooling, MaxPool.Config) else 0.0
        )
        self.lstm = create_module(
            config.lstm, embed_dim=embed_dim, padding_value=padding_value
        )

        lstm_out_dim = self.lstm.representation_dim

        # Document projection and attention.
        self.use_doc_attention = config.pooling is not None
        if config.pooling:
            self.doc_attention = (
                create_module(config.pooling, n_input=lstm_out_dim)
                if config.pooling
                else lambda x: x
            )
            layers = []
            for _ in range(config.doc_mlp_layers - 1):
                layers.extend(
                    [nn.Linear(lstm_out_dim, lstm_out_dim), self.relu, self.dropout]
                )
            layers.append(nn.Linear(lstm_out_dim, lstm_out_dim))
            self.projection_d = nn.Sequential(*layers)

        # Word projection and attention.
        self.use_word_attention = config.slot_attention is not None
        if config.slot_attention:
            word_out_dim = lstm_out_dim
            self.word_attention = lambda x: x
            if config.slot_attention.attention_type != SlotAttentionType.NO_ATTENTION:
                self.word_attention = SlotAttention(
                    config.slot_attention,
                    lstm_out_dim,
                    batch_first=True,
                )
                word_out_dim += lstm_out_dim

            layers = [nn.Linear(word_out_dim, lstm_out_dim), self.relu, self.dropout]
            for _ in range(config.word_mlp_layers - 2):
                layers.extend(
                    [nn.Linear(lstm_out_dim, lstm_out_dim), self.relu, self.dropout]
                )
            layers.append(nn.Linear(lstm_out_dim, lstm_out_dim))
            self.projection_w = nn.Sequential(*layers)

        # Set the representation dimension attribute.
        self.representation_dim = (
            self.doc_representation_dim
        ) = self.word_representation_dim = lstm_out_dim

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        *args,
        states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        # Shared layers
        lstm_output, new_state = self.lstm(embedded_tokens, seq_lengths, states)

        # Default doc representation is hidden state of last cell of LSTM.
        # Default word representation is the output state of each cell of LSTM.
        outputs = [
            new_state[0].contiguous().view(-1, self.doc_representation_dim),
            lstm_output,
        ]
        if self.use_doc_attention:
            outputs[0] = self.projection_d(self.doc_attention(lstm_output))
        if self.use_word_attention:
            outputs[1] = self.projection_w(self.word_attention(lstm_output))

        return outputs[0], outputs[1], new_state
