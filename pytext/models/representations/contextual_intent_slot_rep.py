#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple, Union

import torch
from pytext.models.module import create_module

from .bilstm_doc_slot_attention import BiLSTMDocSlotAttention
from .docnn import DocNNRepresentation
from .jointcnn_rep import JointCNNRepresentation
from .representation_base import RepresentationBase


class ContextualIntentSlotRepresentation(RepresentationBase):
    """
    Representation for a contextual intent slot model

    The inputs are two embeddings: word level embedding containing dictionary features,
    sequence (contexts) level embedding. See following diagram for the representation
    implementation that combines the two embeddings. Seq_representation is concatenated
    with word_embeddings.

    ::

        +-----------+
        | word_embed|--------------------------->+   +--------------------+
        +-----------+                            |   | doc_representation |
        +-----------+   +-------------------+    |-->+--------------------+
        | seq_embed |-->| seq_representation|--->+   | word_representation|
        +-----------+   +-------------------+        +--------------------+
                                                      joint_representation

    """

    class Config(RepresentationBase.Config):
        sen_representation: DocNNRepresentation.Config = DocNNRepresentation.Config()
        seq_representation: DocNNRepresentation.Config = DocNNRepresentation.Config()
        joint_representation: Union[
            BiLSTMDocSlotAttention.Config, JointCNNRepresentation.Config
        ] = BiLSTMDocSlotAttention.Config()

    def __init__(self, config: Config, embed_dim: Tuple[int, ...]) -> None:
        super().__init__(config)
        assert len(embed_dim) == 2
        self.sen_rep = create_module(config.sen_representation, embed_dim=embed_dim[1])
        self.sen_representation_dim = self.sen_rep.representation_dim

        self.seq_rep = create_module(
            config.seq_representation, embed_dim=self.sen_representation_dim
        )
        self.seq_representation_dim = self.seq_rep.representation_dim
        self.joint_rep = create_module(
            config.joint_representation,
            embed_dim=embed_dim[0] + self.seq_representation_dim,
        )
        self.doc_representation_dim = self.joint_rep.doc_representation_dim
        self.word_representation_dim = self.joint_rep.word_representation_dim

    def forward(
        self,
        word_seq_embed: Tuple[torch.Tensor, torch.Tensor],
        word_lengths: torch.Tensor,
        seq_lengths: torch.Tensor,
        *args,
    ) -> List[torch.Tensor]:

        (word_embed, seq_embed) = word_seq_embed

        (bsz, max_num_sen, max_seq_len, dim) = seq_embed.size()
        rep = self.sen_rep(seq_embed.view(bsz * max_num_sen, max_seq_len, dim))
        sentence_reps = rep.view(bsz, max_num_sen, self.sen_representation_dim)
        seq_out = self.seq_rep(embedded_tokens=sentence_reps)

        bsz, max_seq_len, dim = word_embed.size()
        seq_rep_expand = seq_out.view(bsz, 1, -1).repeat(1, max_seq_len, 1)
        new_embed = torch.cat([seq_rep_expand, word_embed], 2)
        return self.joint_rep(new_embed, word_lengths)
