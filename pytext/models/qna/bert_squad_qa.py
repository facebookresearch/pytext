#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Union

import torch
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.data.squad_for_bert_tensorizer import (
    SquadForBERTTensorizer,
    SquadForRoBERTaTensorizer,
)
from pytext.data.tensorizers import LabelTensorizer
from pytext.data.utils import Vocabulary
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import BaseModel
from pytext.models.module import create_module
from pytext.models.output_layers.squad_output_layer import SquadOutputLayer
from pytext.models.representations.huggingface_bert_sentence_encoder import (
    HuggingFaceBertSentenceEncoder,
)
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)


class BertSquadQAModel(NewBertModel):

    __EXPANSIBLE__ = True

    class Config(NewBertModel.Config):
        class ModelInput(BaseModel.Config.ModelInput):
            squad_input: Union[
                SquadForBERTTensorizer.Config, SquadForRoBERTaTensorizer.Config
            ] = SquadForBERTTensorizer.Config(max_seq_len=256)
            # is_impossible label
            has_answer: LabelTensorizer.Config = LabelTensorizer.Config(
                column="has_answer"
            )

        inputs: ModelInput = ModelInput()
        encoder: TransformerSentenceEncoderBase.Config = (
            HuggingFaceBertSentenceEncoder.Config()
        )
        pos_decoder: MLPDecoder.Config = MLPDecoder.Config(out_dim=2)
        has_ans_decoder: MLPDecoder.Config = MLPDecoder.Config(out_dim=2)
        output_layer: SquadOutputLayer.Config = SquadOutputLayer.Config()
        is_kd: bool = False

    @classmethod
    def from_config(cls, config: Config, tensorizers):
        has_answer_labels = ["False", "True"]
        tensorizers["has_answer"].vocab = Vocabulary(has_answer_labels)
        vocab = tensorizers["squad_input"].vocab

        encoder = create_module(
            config.encoder,
            output_encoded_layers=True,
            padding_idx=vocab.get_pad_index(),
            vocab_size=vocab.__len__(),
        )

        pos_decoder = create_module(
            config.pos_decoder, in_dim=encoder.representation_dim, out_dim=2
        )
        has_ans_decoder = create_module(
            config.has_ans_decoder,
            in_dim=encoder.representation_dim,
            out_dim=len(has_answer_labels),
        )

        output_layer = create_module(
            config.output_layer, labels=has_answer_labels, is_kd=config.is_kd
        )

        return cls(
            encoder, pos_decoder, has_ans_decoder, output_layer, is_kd=config.is_kd
        )

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        has_ans_decoder: nn.Module,
        output_layer: nn.Module,
        stage: Stage = Stage.TRAIN,
        is_kd: bool = Config.is_kd,
    ) -> None:
        super().__init__(encoder, decoder, output_layer, stage)
        self.has_ans_decoder = has_ans_decoder
        self.module_list.append(has_ans_decoder)
        self.is_kd = is_kd

    def arrange_model_inputs(self, tensor_dict):
        tokens, pad_mask, segment_labels, positions, *ignore = tensor_dict[
            "squad_input"
        ]
        return tokens, pad_mask, segment_labels, positions

    def arrange_targets(self, tensor_dict):
        # has_answer = True if answer exists
        has_answer = tensor_dict["has_answer"]
        if not self.is_kd:
            _, _, _, _, answer_start_idx, answer_end_idx = tensor_dict["squad_input"]
            return answer_start_idx, answer_end_idx, has_answer
        else:
            (
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

    def forward(self, *inputs):
        tokens, pad_mask, segment_labels, _ = inputs  # See arrange_model_inputs()
        encoded_layers, cls_embed = self.encoder(inputs)
        pos_logits = self.decoder(encoded_layers[-1])
        if isinstance(pos_logits, (list, tuple)):
            pos_logits = pos_logits[0]

        has_answer_logits = (
            torch.zeros((pos_logits.size(0), 2))  # dummy tensor
            if self.output_layer.ignore_impossible
            else self.has_ans_decoder(cls_embed)
        )
        # Shape of logits is (batch_size, seq_len, 2)
        start_logits, end_logits = pos_logits.split(1, dim=-1)

        # Shape of start_logits and end_logits is (batch_size, seq_len, 1)
        # Hence, remove the last dimension and reduce them to the dimensions to
        # (batch_size, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, has_answer_logits, pad_mask, segment_labels
