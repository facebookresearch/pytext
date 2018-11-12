#!/usr/bin/env python3

from typing import List, Union

import torch
from pytext.config import ConfigBase
from pytext.models.module import create_module

from .bilstm_doc_slot_attention import BiLSTMDocSlotAttention
from .jointcnn_rep import JointCNNRepresentation
from .representation_base import RepresentationBase
from .seq_rep import SeqRepresentation


class ContextualIntentSlotRepresentation(RepresentationBase):
    """
    Representation for a contextual intent slot model

    The inputs are two embeddings: word level embedding containing dictionary features,
    sequence (contexts) level embedding. See following diagram for the representation
    implementaion that combine the two embeddings. Seq_representation is concatenated
    with word_embeddings.

    +-----------+
    | word_embed|--------------------------->+   +--------------------+
    +-----------+                            |   | doc_representation |
    +-----------+   +-------------------+    |-->+--------------------+
    | seq_embed |-->| seq_representation|--->+   | word_representation|
    +-----------+   +-------------------+        +--------------------+
                                                   joint_representation
    """

    class Config(RepresentationBase.Config):
        seq_representation: SeqRepresentation.Config = SeqRepresentation.Config()
        joint_representation: Union[
            BiLSTMDocSlotAttention.Config, JointCNNRepresentation.Config
        ] = BiLSTMDocSlotAttention.Config()

    def __init__(self, config: Config, embed_dim: int, seq_embed_dim: int) -> None:
        super().__init__(config)
        self.seq_rep = create_module(config.seq_representation, embed_dim=seq_embed_dim)
        self.seq_representation_dim = self.seq_rep.representation_dim
        self.joint_rep = create_module(
            config.joint_representation,
            embed_dim=embed_dim + self.seq_representation_dim,
        )

    def forward(
        self,
        word_embed: torch.Tensor,
        seq_embed: torch.Tensor,
        word_lengths: torch.Tensor,
        seq_lengths: torch.Tensor,
        *args,
    ) -> List[torch.Tensor]:
        # Every batch is sorted by in descending or of word_lengths.
        # We need to sort seq_lengths and seq_embed first before passing
        # to seq_rep, then unsort the output of seq_rep so it aligns with batch order

        # sort seq_lengths and seq_embed
        seq_lengths, sort_idx = torch.sort(seq_lengths, descending=True)
        _, unsort_idx = torch.sort(sort_idx)
        seq_embed = seq_embed[sort_idx]

        seq_rep = self.seq_rep(embedded_seqs=seq_embed, seq_lengths=seq_lengths)

        # unsort seq_out
        seq_out = seq_rep[0][unsort_idx]

        bsz, max_seq_len, dim = word_embed.size()
        seq_rep_expand = seq_out.view(bsz, 1, -1).expand(-1, max_seq_len, -1)
        new_embed = torch.cat([seq_rep_expand, word_embed], 2)
        return self.joint_rep(new_embed, word_lengths)
