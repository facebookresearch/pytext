#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union

import torch
from pytext.common.constants import Stage
from pytext.data.masked_tensorizer import MaskedTokenTensorizer
from pytext.data.tensorizers import GazetteerTensorizer, Tensorizer, TokenTensorizer
from pytext.data.utils import Vocabulary
from pytext.models.embeddings.scriptable_embedding_list import ScriptableEmbeddingList
from pytext.models.model import Model
from pytext.models.module import create_module

from .conv_model import CNNModel, DecoupledCNNModel
from .mask_generator import MaskedSequenceGenerator
from .nar_length import ConvLengthPredictionModule, MaskedLengthPredictionModule
from .nar_output_layer import NARSeq2SeqOutputLayer
from .seq2seq_model import Seq2SeqModel


def nar_model_forward_aux(
    length_prediction_model,
    output_dict,
    dict_feats=None,
):
    """
    This function prepares the model configs dictionary for the late fusion models. In particular,
    It checks whether a dictionary embedding is specified. If it does, create a dict_embedding, but do not
    include it as part of the scripttable src_mebedding_list. Beside this, the rest of the code is similar
    to that of a MaskDecopuledSeq2Seq model
    """
    encoder_mask: Optional[torch.Tensor] = None
    if "encoder_mask" in output_dict:
        encoder_mask = output_dict["encoder_mask"]
    (
        output_dict["predicted_tgt_lengths"],
        output_dict["predicted_tgt_lengths_logits"],
    ) = length_prediction_model(output_dict["encoder_out"], encoder_mask)
    if dict_feats:
        (
            output_dict["dict_tokens"],
            output_dict["dict_weights"],
            output_dict["dict_lengths"],
        ) = dict_feats

    return output_dict


def create_src_embedding_list(config, tensorizers):
    src_tokens = tensorizers["src_seq_tokens"]
    src_embedding_list = [create_module(config.source_embedding, tensorizer=src_tokens)]
    source_vocab = src_tokens.vocab
    gazetteer_tensorizer = tensorizers.get("dict_feat")
    dict_vocab, dict_embedding = None, None
    if gazetteer_tensorizer:
        dict_embedding = create_module(
            config.dict_embedding, tensorizer=gazetteer_tensorizer
        )
        src_embedding_list.append(dict_embedding)
        dict_vocab = gazetteer_tensorizer.vocab
    source_embedding = ScriptableEmbeddingList(src_embedding_list)
    return source_embedding, source_vocab, dict_embedding, dict_vocab


def create_tgt_embedding_list(config, tensorizers):
    trg_tokens = tensorizers["trg_seq_tokens"]
    target_embedding = ScriptableEmbeddingList(
        [create_module(config.target_embedding, tensorizer=trg_tokens)]
    )
    return target_embedding, trg_tokens.vocab


class NARSeq2SeqModel(Seq2SeqModel):
    class Config(Seq2SeqModel.Config):
        class ModelInput(Model.Config.ModelInput):
            src_seq_tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            trg_seq_tokens: MaskedTokenTensorizer.Config = (
                MaskedTokenTensorizer.Config()
            )
            dict_feat: Optional[GazetteerTensorizer.Config] = None

        encoder_decoder: DecoupledCNNModel.Config = DecoupledCNNModel.Config()
        inputs: ModelInput = ModelInput()
        output_layer: NARSeq2SeqOutputLayer.Config = NARSeq2SeqOutputLayer.Config()
        sequence_generator: MaskedSequenceGenerator.Config = (
            MaskedSequenceGenerator.Config()
        )
        length_prediction_model: Union[
            MaskedLengthPredictionModule.Config, ConvLengthPredictionModule.Config
        ] = MaskedLengthPredictionModule.Config()

    def __init__(
        self,
        model: CNNModel,
        length_prediction_model: MaskedLengthPredictionModule,
        output_layer: NARSeq2SeqOutputLayer,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        dictfeat_vocab: Vocabulary,
        tensorizer=None,
        generator_config=None,
        config: Config = None,
    ):
        super().__init__(
            model,
            output_layer,
            src_vocab,
            trg_vocab,
            dictfeat_vocab,
            generator_config=None,
        )
        self.quantize = generator_config.quantize
        self.length_prediction_model = length_prediction_model
        self.sequence_generator_builder = (
            lambda model, length_prediction_model, quantize: create_module(
                generator_config, model, length_prediction_model, trg_vocab, quantize
            )
        )
        self.force_eval_predictions = generator_config.force_eval_predictions
        self.generate_predictions_every = generator_config.generate_predictions_every
        self.tensorizer = tensorizer

    def get_embeddings(self, embedding_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward(
        self,
        src_tokens: torch.Tensor,
        dict_feats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        src_lengths: torch.Tensor,
        trg_tokens: torch.Tensor,
        src_subword_begin_indices: Optional[torch.Tensor] = None,
    ):
        additional_features: List[List[torch.Tensor]] = []

        if dict_feats is not None:
            additional_features.append(list(dict_feats))

        logits, output_dict = self.model(
            src_tokens=src_tokens,
            additional_features=additional_features,
            src_lengths=src_lengths,
            prev_output_tokens=trg_tokens,
            src_subword_begin_indices=src_subword_begin_indices,
        )

        output_dict = nar_model_forward_aux(
            self.length_prediction_model,
            output_dict,
            dict_feats=dict_feats,
        )

        return logits, output_dict

    def get_pred(self, model_outputs, context=None):
        predictions, scores = (None, None)
        if context:
            stage = context.get("stage", None)
            if stage and (
                (stage == Stage.TEST)
                or (
                    self.force_eval_predictions
                    and stage == Stage.EVAL
                    and (context["epoch"] % self.generate_predictions_every == 0)
                )
            ):
                if self.sequence_generator is None:
                    assert not self.model.training
                    # During evaluation, quantize is set to False or it will
                    # create with issues in evaluating the model on gpu
                    self.sequence_generator = self.sequence_generator_builder(
                        self.model, self.length_prediction_model, False
                    )
                _, model_input = model_outputs
                predictions, scores = self.sequence_generator.generate_hypo(model_input)
        return predictions, scores

    @classmethod
    def train_batch(cls, model, batch, state=None):
        # Forward pass through the network.
        model_inputs = model.arrange_model_inputs(batch)
        model_context = model.arrange_model_context(batch)
        targets = model.arrange_targets(batch)
        model_outputs = model(*model_inputs)

        # Add stage to context.
        if state:
            if model_context is None:
                model_context = {"stage": state.stage, "epoch": state.epoch}
            else:
                model_context["stage"] = state.stage
                model_context["epoch"] = state.epoch

        # Compute loss and predictions.
        loss, loss_dict = model.get_loss(model_outputs, targets, model_context)
        predictions, scores = model.get_pred(model_outputs, context=model_context)

        # Pack results and return them.
        metric_data = (predictions, targets, scores, (loss, loss_dict), model_inputs)
        return loss, metric_data

    def arrange_model_inputs(self, tensor_dict):
        src_tokens, src_lengths, _ = tensor_dict["src_seq_tokens"]
        (
            trg_tokens,
            trg_lengths,
            _,
            masked_trg_tokens_source,
            _masked_trg_tokens_target,
        ) = tensor_dict["trg_seq_tokens"]

        return (
            src_tokens,
            tensor_dict.get("dict_feat"),
            src_lengths,
            # this trg_tokens is used for teacher forcing
            masked_trg_tokens_source,
        )

    def arrange_targets(self, tensor_dict):
        (
            trg_tokens,
            trg_lengths,
            _,
            masked_trg_tokens_source,
            masked_trg_tokens_target,
        ) = tensor_dict["trg_seq_tokens"]
        return (trg_tokens, masked_trg_tokens_target), trg_lengths

    def state_dict(self, *args, **kwargs):
        # If we dont set to None, it will create issues with
        # loading snapshots
        self.sequence_generator = None
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # If we dont set to None, it will create issues with
        # loading snapshots
        self.sequence_generator = None
        super().load_state_dict(state_dict, *args, **kwargs)

    @classmethod
    def construct_length_prediction_module(
        cls,
        config: Config,
    ):
        return create_module(
            config.length_prediction_model,
            config.encoder_decoder.encoder.encoder_config.encoder_embed_dim,
        )

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        (
            source_embedding,
            source_vocab,
            dict_embedding,
            dict_vocab,
        ) = create_src_embedding_list(config, tensorizers)
        target_embedding, target_vocab = create_tgt_embedding_list(config, tensorizers)
        model = create_module(
            config.encoder_decoder,
            source_vocab,
            source_embedding,
            target_vocab,
            target_embedding,
            dict_embedding=dict_embedding,
        )
        output_layer = create_module(config.output_layer, target_vocab)
        length_prediction = cls.construct_length_prediction_module(config)
        return cls(
            model=model,
            length_prediction_model=length_prediction,
            output_layer=output_layer,
            src_vocab=source_vocab,
            trg_vocab=target_vocab,
            dictfeat_vocab=dict_vocab if tensorizers.get("dict_feat") else None,
            generator_config=config.sequence_generator,
            config=config,
            tensorizer=tensorizers["src_seq_tokens"],
        )


class NARDecoupledSeq2SeqModel(NARSeq2SeqModel):
    class Config(NARSeq2SeqModel.Config):
        encoder_decoder: DecoupledCNNModel.Config = DecoupledCNNModel.Config()
