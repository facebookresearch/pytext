#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Callable, Dict, List, Optional, Tuple

import torch
from pytext.common.constants import Stage
from pytext.data.tensorizers import (
    ByteTokenTensorizer,
    GazetteerTensorizer,
    Tensorizer,
    TokenTensorizer,
)
from pytext.data.utils import Vocabulary
from pytext.models.embeddings import (
    ContextualTokenEmbedding,
    DictEmbedding,
    WordEmbedding,
)
from pytext.models.embeddings.scriptable_embedding_list import ScriptableEmbeddingList
from pytext.models.model import BaseModel, Model
from pytext.models.module import create_module
from pytext.torchscript.seq2seq.export_model import Seq2SeqJIT
from pytext.torchscript.seq2seq.scripted_seq2seq_generator import (
    ScriptedSequenceGenerator,
)
from pytext.utils.cuda import GetTensor
from pytext.utils.usage import log_class_usage

from .rnn_encoder_decoder import RNNModel
from .seq2seq_output_layer import Seq2SeqOutputLayer


class Seq2SeqModel(Model):
    """
    Sequence to sequence model using an encoder-decoder architecture.
    """

    class Config(Model.Config):
        class ModelInput(Model.Config.ModelInput):
            src_seq_tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            trg_seq_tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            dict_feat: Optional[GazetteerTensorizer.Config] = None
            contextual_token_embedding: Optional[ByteTokenTensorizer.Config] = None

        inputs: ModelInput = ModelInput()
        encoder_decoder: RNNModel.Config = RNNModel.Config()
        source_embedding: WordEmbedding.Config = WordEmbedding.Config()
        target_embedding: WordEmbedding.Config = WordEmbedding.Config()
        dict_embedding: Optional[DictEmbedding.Config] = None
        contextual_token_embedding: Optional[ContextualTokenEmbedding.Config] = None
        output_layer: Seq2SeqOutputLayer.Config = Seq2SeqOutputLayer.Config()
        sequence_generator: ScriptedSequenceGenerator.Config = (
            ScriptedSequenceGenerator.Config()
        )

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        src_tokens = tensorizers["src_seq_tokens"]
        src_embedding_list = [
            create_module(config.source_embedding, tensorizer=src_tokens)
        ]
        gazetteer_tensorizer = tensorizers.get("dict_feat")
        if gazetteer_tensorizer:
            src_embedding_list.append(
                create_module(config.dict_embedding, tensorizer=gazetteer_tensorizer)
            )
        contextual_token_tensorizer = tensorizers.get("contextual_token_embedding")
        if contextual_token_tensorizer:
            src_embedding_list.append(
                create_module(
                    config.contextual_token_embedding,
                    tensorizer=contextual_token_tensorizer,
                )
            )
        source_embedding = ScriptableEmbeddingList(src_embedding_list)

        trg_tokens = tensorizers["trg_seq_tokens"]
        target_embedding = ScriptableEmbeddingList(
            [create_module(config.target_embedding, tensorizer=trg_tokens)]
        )

        model = create_module(
            config.encoder_decoder,
            src_tokens.vocab,
            source_embedding,
            trg_tokens.vocab,
            target_embedding,
        )
        output_layer = create_module(config.output_layer, trg_tokens.vocab)

        dictfeat_tokens = tensorizers.get("dict_feat")

        return cls(
            model=model,
            output_layer=output_layer,
            src_vocab=src_tokens.vocab,
            trg_vocab=trg_tokens.vocab,
            dictfeat_vocab=dictfeat_tokens.vocab if dictfeat_tokens else None,
            generator_config=config.sequence_generator,
        )

    def arrange_model_inputs(
        self, tensor_dict
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        src_tokens, src_lengths, _ = tensor_dict["src_seq_tokens"]
        trg_tokens, trg_lengths, _ = tensor_dict["trg_seq_tokens"]

        def _shift_target(in_sequences, seq_lens, eos_idx, pad_idx):
            shifted_sequence = GetTensor(
                torch.LongTensor(in_sequences.size()).fill_(pad_idx)
            )
            for i, in_seq in enumerate(in_sequences):
                shifted_sequence[i, 0] = eos_idx
                # Copy everything except ones starting from the EOS at the end.
                shifted_sequence[i, 1 : seq_lens[i].item()] = in_seq[
                    0 : seq_lens[i].item() - 1
                ]
            return shifted_sequence

        # shift target
        trg_tokens = _shift_target(
            trg_tokens, trg_lengths, self.trg_eos_index, self.trg_pad_index
        )

        return (
            src_tokens,
            tensor_dict.get("dict_feat"),
            tensor_dict.get("contextual_token_embedding"),
            src_lengths,
            trg_tokens,
        )

    def arrange_targets(self, tensor_dict):
        trg_tokens, trg_lengths, _ = tensor_dict["trg_seq_tokens"]
        return (trg_tokens, trg_lengths)

    def __init__(
        self,
        model: RNNModel,
        output_layer: Seq2SeqOutputLayer,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        dictfeat_vocab: Vocabulary,
        generator_config=None,
    ):
        BaseModel.__init__(self)
        self.model = model
        self.output_layer = output_layer

        # Sequence generation is expected to be used only for inference, and to
        # take the trained model(s) as input. Creating the sequence generator
        # may apply Torchscript JIT compilation and quantization, which modify
        # the input model. Therefore, we want to create the sequence generator
        # after training.
        if generator_config is not None:
            self.sequence_generator_builder = lambda models: create_module(
                generator_config, models, trg_vocab.get_eos_index()
            )
        self.sequence_generator = None

        # Disable predictions until testing (see above comment about sequence
        # generator). If this functionality is needed, a new sequence generator
        # with a copy of the model should be used for each epoch during the
        # EVAL stage.
        self.force_eval_predictions = False

        # Target vocab EOS index is useful for recognizing when to stop generating
        self.trg_eos_index = trg_vocab.get_eos_index()

        # Target vocab PAD index is useful for shifting source/target prior to decoding
        self.trg_pad_index = trg_vocab.get_pad_index()

        # Source, target and dictfeat vocab are needed for export so that we can handle
        # string input
        self.src_dict = src_vocab
        self.trg_dict = trg_vocab
        self.dictfeat_dict = dictfeat_vocab

        log_class_usage(__class__)

    def max_decoder_positions(self):
        return self.model.max_decoder_positions()

    def forward(
        self,
        src_tokens: torch.Tensor,
        dict_feats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        contextual_token_embedding: Optional[torch.Tensor],
        src_lengths: torch.Tensor,
        trg_tokens: torch.Tensor,
    ):
        additional_features: List[List[torch.Tensor]] = []

        if dict_feats:
            additional_features.append(list(dict_feats))

        if contextual_token_embedding is not None:
            additional_features.append([contextual_token_embedding])

        logits, output_dict = self.model(
            src_tokens, additional_features, src_lengths, trg_tokens
        )

        if dict_feats:
            (
                output_dict["dict_tokens"],
                output_dict["dict_weights"],
                output_dict["dict_lengths"],
            ) = dict_feats

        if contextual_token_embedding is not None:
            output_dict["contextual_token_embedding"] = contextual_token_embedding

        return logits, output_dict

    def get_pred(self, model_outputs, context=None):
        preds = (None, None)
        if context:
            stage = context.get("stage", None)
            if stage and (
                (stage == Stage.TEST)
                or (self.force_eval_predictions and stage == Stage.EVAL)
            ):
                # We don't support predictions during EVAL since sequence
                # generator may quantize the models.
                assert (
                    not self.force_eval_predictions
                ), "Eval predictions not supported for Seq2SeqModel yet."
                assert self.sequence_generator_builder is not None
                if self.sequence_generator is None:
                    assert not self.model.training
                    self.sequence_generator = torch.jit.script(
                        self.sequence_generator_builder([self.model])
                    )
                _, model_input = model_outputs
                preds = self.sequence_generator.generate_hypo(model_input)
        return preds

    def torchscriptify(self):
        self.model.zero_grad()
        self.model.eval()
        assert self.sequence_generator_builder is not None
        if self.sequence_generator is None:
            self.sequence_generator = self.sequence_generator_builder([self.model])

        model = Seq2SeqJIT(
            self.src_dict,
            self.trg_dict,
            self.sequence_generator,
            filter_eos_bos=True,
            copy_unk_token=True,
        )
        return torch.jit.script(model)
