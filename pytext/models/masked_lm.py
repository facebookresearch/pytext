#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List

import torch
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from pytext.common.constants import SpecialTokens, Stage
from pytext.config import ConfigBase
from pytext.data.bert_tensorizer import BERTTensorizerBase
from pytext.data.tensorizers import Tensorizer
from pytext.data.utils import Vocabulary
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.masking_utils import (
    MaskingStrategy,
    frequency_based_masking,
    random_masking,
)
from pytext.models.model import BaseModel
from pytext.models.module import create_module
from pytext.models.output_layers.lm_output_layer import LMOutputLayer
from pytext.models.representations.transformer_sentence_encoder import (
    TransformerSentenceEncoder,
)
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)
from pytext.utils.usage import log_class_usage


class MaskedLanguageModel(BaseModel):
    """Masked language model for BERT style pre-training."""

    SUPPORT_FP16_OPTIMIZER = True

    class Config(BaseModel.Config):
        class InputConfig(ConfigBase):
            tokens: BERTTensorizerBase.Config = BERTTensorizerBase.Config(
                max_seq_len=128
            )

        inputs: InputConfig = InputConfig()
        encoder: TransformerSentenceEncoderBase.Config = TransformerSentenceEncoder.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: LMOutputLayer.Config = LMOutputLayer.Config()
        mask_prob: float = 0.15
        mask_bos: bool = False
        # masking
        masking_strategy: MaskingStrategy = MaskingStrategy.RANDOM
        # tie weights determines whether the input embedding weights are used
        # in the output vocabulary projection as well
        tie_weights: bool = True

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        token_tensorizer = tensorizers["tokens"]
        vocab = token_tensorizer.vocab

        encoder = create_module(
            config.encoder,
            output_encoded_layers=True,
            padding_idx=vocab.get_pad_index(),
            vocab_size=vocab.__len__(),
        )
        decoder = create_module(
            config.decoder, in_dim=encoder.representation_dim, out_dim=len(vocab)
        )
        if getattr(config.encoder, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        # if weights are not shared then we need to ensure that the decoder
        # params are initialized in the same was as the encoder params
        if config.tie_weights:
            list(decoder.mlp.modules())[-1].weight = encoder._embedding().weight

        output_layer = create_module(config.output_layer, labels=vocab)
        return cls(
            encoder,
            decoder,
            output_layer,
            token_tensorizer,
            vocab,
            mask_prob=config.mask_prob,
            mask_bos=config.mask_bos,
            masking_strategy=config.masking_strategy,
        )

    def __init__(
        self,
        encoder: TransformerSentenceEncoderBase,
        decoder: MLPDecoder,
        output_layer: LMOutputLayer,
        token_tensorizer: BERTTensorizerBase,
        vocab: Vocabulary,
        mask_prob: float = Config.mask_prob,
        mask_bos: float = Config.mask_bos,
        masking_strategy: MaskingStrategy = Config.masking_strategy,
        stage: Stage = Stage.TRAIN,
    ) -> None:
        super().__init__(stage=stage)
        self.encoder = encoder
        self.decoder = decoder
        self.module_list = [encoder, decoder]
        self.output_layer = output_layer
        self.token_tensorizer = token_tensorizer
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.mask_bos = mask_bos
        self.stage = stage
        self.masking_strategy = masking_strategy

        # initialize the frequency based sampling weights if these will be used
        self.token_sampling_weights = None
        if self.masking_strategy == MaskingStrategy.FREQUENCY:
            self.token_sampling_weights = [x ** -0.5 for x in self.vocab.counts]

            # Set probability of masking special tokens to be very low, since it doesn't
            # make sense to use them for MLM (unless there are no other tokens in the
            # the batch).
            tokens_to_avoid_masking = [
                SpecialTokens.PAD,
                SpecialTokens.UNK,
                SpecialTokens.MASK,
            ]
            if not self.mask_bos:
                tokens_to_avoid_masking.extend([SpecialTokens.BOS, SpecialTokens.EOS])
            for token in tokens_to_avoid_masking:
                token_idx = self.vocab.idx.get(token)
                if token_idx is not None:
                    self.token_sampling_weights[token_idx] = 1e-20
        log_class_usage(__class__)

    def arrange_model_inputs(self, tensor_dict):
        tokens, *other = tensor_dict["tokens"]
        self.mask, self.pad_mask, mask_mask, rand_mask = self._get_mask(tokens)
        masked_tokens = self._mask_input(
            tokens, mask_mask, self.vocab.idx[SpecialTokens.MASK]
        )
        masked_tokens = self._mask_input(
            masked_tokens, rand_mask, torch.randint_like(tokens, high=len(self.vocab))
        )
        return (masked_tokens,) + tuple(other)

    def arrange_targets(self, tensor_dict):
        tokens, *other = tensor_dict["tokens"]
        masked_target = self._mask_output(tokens, self.mask)
        # (masked targets, #predicted tokens, #input tokens)
        return masked_target, self.mask.sum(-1), self.pad_mask.sum(-1)

    def forward(self, *inputs) -> List[torch.Tensor]:
        encoded_layers, _ = self.encoder(inputs)
        return self.decoder(encoded_layers[-1][self.mask.bool(), :])

    def _select_tokens_to_mask(
        self, tokens: torch.Tensor, mask_prob: float
    ) -> torch.tensor:
        if self.masking_strategy == MaskingStrategy.RANDOM:
            mask = random_masking(tokens, mask_prob)
            if not self.mask_bos:
                bos_idx = self.vocab.idx[self.token_tensorizer.bos_token]
                mask *= (tokens != bos_idx).long()
            return mask
        elif self.masking_strategy == MaskingStrategy.FREQUENCY:
            return frequency_based_masking(
                tokens, self.token_sampling_weights, mask_prob
            )
        else:
            raise NotImplementedError(
                "Specified Masking Strategy isnt currently implemented."
            )

    def _get_mask(self, tokens):
        mask = self._select_tokens_to_mask(tokens, self.mask_prob)
        pad_mask = (tokens != self.vocab.get_pad_index()).long()
        mask *= pad_mask
        if not mask.byte().any():
            # Keep one masked token to avoid failure in the loss calculation.
            mask[0, 0] = 1

        probs = torch.rand_like(tokens, dtype=torch.float)
        rand_mask = (probs < 0.1).long() * mask
        mask_mask = (probs >= 0.2).long() * mask
        return mask, pad_mask, mask_mask, rand_mask

    def _mask_input(self, tokens, mask, replacement):
        return tokens * (1 - mask) + replacement * mask

    def _mask_output(self, tokens, mask):
        return torch.masked_select(tokens, mask.bool())
