#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.data.squad_tensorizer import SquadTensorizer
from pytext.data.tensorizers import LabelTensorizer, Tensorizer
from pytext.data.utils import Vocabulary
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import WordEmbedding
from pytext.models.model import BaseModel
from pytext.models.module import create_module
from pytext.models.output_layers.squad_output_layer import SquadOutputLayer
from pytext.models.representations.attention import (
    DotProductSelfAttention,
    MultiplicativeAttention,
    SequenceAlignedAttention,
)
from pytext.models.representations.pooling import SelfAttention
from pytext.models.representations.stacked_bidirectional_rnn import (
    StackedBidirectionalRNN,
)


GLOVE_840B_300D = "/mnt/vol/pytext/users/kushall/pretrained/glove.840B.300d.txt"


class DrQAModel(BaseModel):
    class Config(BaseModel.Config):
        class ModelInput(BaseModel.Config.ModelInput):
            squad_input: SquadTensorizer.Config = SquadTensorizer.Config()
            has_answer: LabelTensorizer.Config = LabelTensorizer.Config(
                column="has_answer"
            )

        # Model inputs.
        inputs: ModelInput = ModelInput()

        # Configrable modules for the model.
        dropout: float = 0.4  # Overrides dropout in sub-modules of the model.
        embedding: WordEmbedding.Config = WordEmbedding.Config(
            embed_dim=300,
            pretrained_embeddings_path=GLOVE_840B_300D,
            vocab_from_pretrained_embeddings=True,
        )
        ques_rnn: StackedBidirectionalRNN.Config = StackedBidirectionalRNN.Config(
            dropout=dropout
        )
        doc_rnn: StackedBidirectionalRNN.Config = StackedBidirectionalRNN.Config(
            dropout=dropout
        )

        # Output layer.
        output_layer: SquadOutputLayer.Config = SquadOutputLayer.Config()

        # Is model traning distilling knowledg from a teacher model?
        is_kd: bool = False

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        # Although the RNN params are configurable, for DrQA we want to set
        # the following parameters for all cases.
        config.ques_rnn.dropout = config.dropout
        config.doc_rnn.dropout = config.dropout

        embedding = cls.create_embedding(config, tensorizers)
        ques_aligned_doc_attn = SequenceAlignedAttention(embedding.embedding_dim)
        ques_rnn = create_module(config.ques_rnn, input_size=embedding.embedding_dim)
        doc_rnn = create_module(config.doc_rnn, input_size=embedding.embedding_dim * 2)
        ques_self_attn = DotProductSelfAttention(ques_rnn.representation_dim)
        start_attn = MultiplicativeAttention(
            doc_rnn.representation_dim, ques_rnn.representation_dim, normalize=False
        )
        end_attn = MultiplicativeAttention(
            doc_rnn.representation_dim, ques_rnn.representation_dim, normalize=False
        )
        doc_rep_pool = SelfAttention(
            SelfAttention.Config(dropout=config.dropout),
            n_input=doc_rnn.representation_dim,
        )
        has_answer_labels = ["False", "True"]
        tensorizers["has_answer"].vocab = Vocabulary(has_answer_labels)
        has_ans_decoder = MLPDecoder(
            config=MLPDecoder.Config(),
            in_dim=doc_rnn.representation_dim,
            out_dim=len(has_answer_labels),
        )
        output_layer = create_module(
            config.output_layer, labels=has_answer_labels, is_kd=config.is_kd
        )
        return cls(
            dropout=nn.Dropout(config.dropout),
            embedding=embedding,
            ques_rnn=ques_rnn,
            doc_rnn=doc_rnn,
            ques_self_attn=ques_self_attn,
            ques_aligned_doc_attn=ques_aligned_doc_attn,
            start_attn=start_attn,
            end_attn=end_attn,
            doc_rep_pool=doc_rep_pool,
            has_ans_decoder=has_ans_decoder,
            output_layer=output_layer,
            is_kd=config.is_kd,
        )

    @classmethod
    def create_embedding(cls, model_config: Config, tensorizers: Dict[str, Tensorizer]):
        squad_tensorizer = tensorizers["squad_input"]

        # Initialize the embedding module.
        embedding_module = create_module(model_config.embedding, None, squad_tensorizer)

        # Set ques and doc tensorizer vocab to squad_tensorizer.vocab.
        squad_tensorizer.ques_tensorizer.vocab = squad_tensorizer.vocab
        squad_tensorizer.doc_tensorizer.vocab = squad_tensorizer.vocab

        return embedding_module

    def __init__(
        self,
        dropout: nn.Module,
        embedding: nn.Module,
        ques_rnn: nn.Module,
        doc_rnn: nn.Module,
        ques_self_attn: nn.Module,
        ques_aligned_doc_attn: nn.Module,
        start_attn: nn.Module,
        end_attn: nn.Module,
        doc_rep_pool: nn.Module,
        has_ans_decoder: nn.Module,
        output_layer: nn.Module,
        is_kd: bool = Config.is_kd,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.embedding = embedding
        self.ques_rnn = ques_rnn
        self.doc_rnn = doc_rnn
        self.ques_self_attn = ques_self_attn
        self.ques_aligned_doc_attn = ques_aligned_doc_attn
        self.start_attn = start_attn
        self.end_attn = end_attn
        self.doc_rep_pool = doc_rep_pool
        self.has_ans_decoder = has_ans_decoder
        self.output_layer = output_layer
        self.ignore_impossible = output_layer.ignore_impossible
        self.module_list = [
            embedding,
            ques_rnn,
            doc_rnn,
            ques_self_attn,
            ques_aligned_doc_attn,
            start_attn,
            end_attn,
            has_ans_decoder,
        ]
        self.is_kd = is_kd

    def arrange_model_inputs(self, tensor_dict):
        (
            doc_tokens,
            doc_seq_len,
            doc_mask,
            ques_tokens,
            ques_seq_len,
            ques_mask,
            *ignore,
        ) = tensor_dict["squad_input"]
        return (doc_tokens, doc_seq_len, doc_mask, ques_tokens, ques_seq_len, ques_mask)

    def arrange_targets(self, tensor_dict):
        has_answer = tensor_dict["has_answer"]
        if not self.is_kd:
            _, _, _, _, _, _, answer_start_idx, answer_end_idx = tensor_dict[
                "squad_input"
            ]
            return answer_start_idx, answer_end_idx, has_answer
        else:
            (
                _,
                _,
                _,
                _,
                _,
                _,
                answer_start_idx,
                answer_end_idx,
                start_logits,
                end_logits,
                has_answer_logits,
            ) = tensor_dict["squad_input"]
            return (
                answer_start_idx,
                answer_end_idx,
                has_answer,
                start_logits,
                end_logits,
                has_answer_logits,
            )

    def forward(
        self,
        doc_tokens: torch.Tensor,
        doc_seq_len: torch.Tensor,
        doc_mask: torch.Tensor,
        ques_tokens: torch.Tensor,
        ques_seq_len: torch.Tensor,
        ques_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Embedding lookups.
        doc_embedded = self.dropout(self.embedding(doc_tokens))
        ques_embedded = self.dropout(self.embedding(ques_tokens))

        # Document encoding.
        doc_attn_weights = self.ques_aligned_doc_attn(
            doc_embedded, ques_embedded, ques_mask
        )
        ques_aligned_doc_embedded = doc_attn_weights.bmm(ques_embedded)
        doc_embedded = torch.cat([doc_embedded, ques_aligned_doc_embedded], dim=2)
        doc_seq_vec = self.doc_rnn(doc_embedded, doc_mask)

        # Question encoding.
        ques_seq_vec = self.ques_rnn(ques_embedded, ques_mask)
        ques_attn_weights = self.ques_self_attn(ques_seq_vec, ques_mask)
        ques_vec = ques_attn_weights.unsqueeze(1).bmm(ques_seq_vec).squeeze(1)

        # Apply dropout to the ques and document representations.
        doc_seq_vec = self.dropout(doc_seq_vec)
        ques_vec = self.dropout(ques_vec)

        # Compute bilinear attention weights for each document token.
        # These attention weights serve as logits for detecting start and end.
        start_logits = self.start_attn(doc_seq_vec, ques_vec, doc_mask)
        end_logits = self.end_attn(doc_seq_vec, ques_vec, doc_mask)

        has_answer_logits = torch.zeros(
            ques_tokens.size(0),  # batch_size
            self.has_ans_decoder.out_dim,  # Number of classes
        )
        if not self.ignore_impossible:  # Compute whether document has an answer.
            doc_vec = self.doc_rep_pool(doc_seq_vec, doc_seq_len)
            has_answer_logits = F.relu(self.has_ans_decoder(doc_vec))

        # start_logits and end_logits: batch_size, max_seq_len
        # has_answer_logit: batch_size, 2
        return start_logits, end_logits, has_answer_logits
