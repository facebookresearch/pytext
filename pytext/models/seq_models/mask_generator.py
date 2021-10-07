#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
from pytext.config import ConfigBase
from pytext.models.module import Module
from pytext.torchscript.vocab import ScriptVocabulary
from torch import Tensor
from torch.quantization import float_qparams_weight_only_qconfig


class BeamRankingAlgorithm(Enum):

    LENGTH_CONDITIONED_RANK: str = "LENGTH_CONDITIONED_RANK"
    LENGTH_CONDITIONED_RANK_MUL: str = "LENGTH_CONDITIONED_RANK_MUL"
    AVERAGE_TOKEN_LPROB: str = "AVERAGE_TOKEN_LPROB"
    TOKEN_LPROB: str = "TOKEN_LPROB"
    LENGTH_CONDITIONED_AVERAGE_TOKEN_LPROB: str = (
        "LENGTH_CONDITIONED_AVERAGE_TOKEN_LPROB"
    )
    LENGTH_CONDITIONED_AVERAGE_TOKEN_LPROB_MULTIPLIED: str = (
        "LENGTH_CONDITIONED_AVERAGE_TOKEN_LPROB_MULTIPLIED"
    )
    LEN_ONLY: str = "LEN_ONLY"


# Sum of token prob and length prob
def length_conditioned_rank(
    token_lprob: torch.Tensor, length_lprob: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    return token_lprob + length_lprob


# Sum of token prob + length * length_prob
def length_conditioned_rank_mul(
    token_lprob: torch.Tensor, length_lprob: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    return token_lprob + target_lengths * length_lprob


# Sum of token prod
def token_prob(
    token_lprob: torch.Tensor, length_lprob: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    return token_lprob


# Only length_prob
def length(
    token_lprob: torch.Tensor, length_lprob: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    return length_lprob


# Avg token prob
def avg_token_lprob(
    token_lprob: torch.Tensor, length_lprob: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    avg_log_prob = token_lprob / target_lengths.to(token_lprob.dtype)
    return avg_log_prob


# Avg token prob + length prob
def length_conditioned_avg_lprob_rank(
    token_lprob: torch.Tensor, length_lprob: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    avg_token_lprob_tensor = avg_token_lprob(token_lprob, length_lprob, target_lengths)
    return avg_token_lprob_tensor + length_lprob


# Avg token prob + len * length_prob
def length_conditioned_avg_lprob_rank_mul(
    token_lprob: torch.Tensor, length_lprob: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    avg_token_lprob_tensor = avg_token_lprob(token_lprob, length_lprob, target_lengths)
    return avg_token_lprob_tensor + target_lengths * length_lprob


def get_beam_ranking_function(ranking_algorithm: BeamRankingAlgorithm):
    if ranking_algorithm == BeamRankingAlgorithm.LENGTH_CONDITIONED_RANK:
        return length_conditioned_rank
    elif ranking_algorithm == BeamRankingAlgorithm.LENGTH_CONDITIONED_RANK_MUL:
        return length_conditioned_rank_mul
    elif ranking_algorithm == BeamRankingAlgorithm.AVERAGE_TOKEN_LPROB:
        return avg_token_lprob
    elif (
        ranking_algorithm == BeamRankingAlgorithm.LENGTH_CONDITIONED_AVERAGE_TOKEN_LPROB
    ):
        return length_conditioned_avg_lprob_rank
    elif (
        ranking_algorithm
        == BeamRankingAlgorithm.LENGTH_CONDITIONED_AVERAGE_TOKEN_LPROB_MULTIPLIED
    ):
        return length_conditioned_avg_lprob_rank_mul
    elif ranking_algorithm == BeamRankingAlgorithm.TOKEN_LPROB:
        return token_prob
    elif ranking_algorithm == BeamRankingAlgorithm.LEN_ONLY:
        return length
    else:
        raise Exception(f"Unknown ranking algorithm {ranking_algorithm}")


def prepare_masked_target_for_lengths(
    beam: Tensor, mask_idx: int, pad_idx: int, length_beam_size: int = 1
) -> Tuple[Tensor, Tensor]:
    # beam : bsz X beam_size
    max_len = beam.max().item()
    bsz = beam.size(0)
    # length_mask[sample_length] will give a row vector of sample_length+1 ones
    # and rest zeros
    length_mask = torch.triu(
        torch.ones(max_len, max_len, device=beam.device, dtype=beam.dtype),
        diagonal=1,
    )
    beam_indices = beam - 1
    length_mask = length_mask[beam_indices.reshape(-1)].reshape(
        bsz, length_beam_size, max_len
    )
    tgt_tokens = torch.zeros(
        bsz, length_beam_size, max_len, device=beam.device, dtype=beam.dtype
    ).fill_(mask_idx)
    tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * pad_idx
    tgt_tokens = tgt_tokens.view(bsz * length_beam_size, max_len)
    return tgt_tokens, length_mask


class EmbedQuantizeType(Enum):
    BIT_8 = "8bit"
    BIT_4 = "4bit"
    NONE = "None"


class MaskedSequenceGenerator(Module):
    class Config(ConfigBase):
        beam_size: int = 3
        quantize: bool = True
        embed_quantize: EmbedQuantizeType = EmbedQuantizeType.NONE
        use_gold_length: bool = False
        force_eval_predictions: bool = True
        generate_predictions_every: int = 1
        beam_ranking_algorithm: BeamRankingAlgorithm = (
            BeamRankingAlgorithm.LENGTH_CONDITIONED_RANK
        )
        clip_target_length: bool = False
        # We use a quardratic formula to generate the max target length
        #   min(targetlen_cap, targetlen_a*x^2 + targetlen_b*x + targetlen_c)
        targetlen_cap: int = 30
        targetlen_a: float = 0
        targetlen_b: float = 2
        targetlen_c: float = 2

    @classmethod
    def from_config(
        cls,
        config,
        model,
        length_prediction,
        trg_vocab,
        quantize=False,
        embed_quantize=False,
    ):
        return cls(
            config,
            model,
            length_prediction,
            trg_vocab,
            config.beam_size,
            config.use_gold_length,
            config.beam_ranking_algorithm,
            quantize,
            config.embed_quantize,
        )

    def __init__(
        self,
        config,
        model,
        length_prediction_model,
        trg_vocab,
        beam_size,
        use_gold_length,
        beam_ranking_algorithm,
        quantize,
        embed_quantize,
    ):
        super().__init__()
        length_prediction_model = length_prediction_model.create_eval_module()
        if quantize:
            qconfig_dict = {
                torch.nn.Linear: torch.quantization.per_channel_dynamic_qconfig
            }
            # embedding quantization
            if embed_quantize != EmbedQuantizeType.NONE:

                # 8-bit embedding quantization
                if embed_quantize == EmbedQuantizeType.BIT_8:
                    qconfig_dict[torch.nn.Embedding] = float_qparams_weight_only_qconfig

                # 4-bit embedding quantization
                elif embed_quantize == EmbedQuantizeType.BIT_4:
                    raise NotImplementedError(
                        "4bit embedding quantization not yet supported"
                    )
                else:
                    raise NotImplementedError(
                        "Embedding Quantization should be either 8bit or 4bit"
                    )

            self.model = torch.quantization.quantize_dynamic(
                model,
                qconfig_dict,
                dtype=torch.qint8,
                inplace=False,
            )

            self.length_prediction_model = torch.quantization.quantize_dynamic(
                length_prediction_model,
                {torch.nn.Linear: torch.quantization.per_channel_dynamic_qconfig},
                dtype=torch.qint8,
                inplace=False,
            )
        else:
            self.model = model
            self.length_prediction_model = length_prediction_model

        self.trg_vocab = ScriptVocabulary(
            list(trg_vocab),
            pad_idx=trg_vocab.get_pad_index(),
            bos_idx=trg_vocab.get_bos_index(-1),
            eos_idx=trg_vocab.get_eos_index(-1),
            mask_idx=trg_vocab.get_mask_index(),
        )
        self.length_beam_size = beam_size
        self.use_gold_length = use_gold_length
        self.beam_ranking_algorithm = get_beam_ranking_function(
            ranking_algorithm=beam_ranking_algorithm
        )
        self.clip_target_length = config.clip_target_length
        self.targetlen_cap = config.targetlen_cap
        self.targetlen_a = config.targetlen_a
        self.targetlen_b = config.targetlen_b
        self.targetlen_c = config.targetlen_c

    def get_encoder_out(
        self,
        src_tokens: Tensor,
        dict_feats: Optional[Tuple[Tensor, Tensor, Tensor]],
        contextual_embed: Optional[Tensor],
        char_feats: Optional[Tensor],
        src_subword_begin_indices: Optional[Tensor],
        src_lengths: Tensor,
        src_index_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        embedding_input = [[src_tokens]]
        if dict_feats is not None:
            embedding_input.append(list(dict_feats))
        if contextual_embed is not None:
            embedding_input.append([contextual_embed])
        if char_feats is not None:
            embedding_input.append([char_feats])
        embeddings = self.model.source_embeddings(embedding_input)
        encoder_out = self.model.encoder(
            src_tokens, embeddings, src_lengths=src_lengths
        )

        if src_index_tokens is not None:
            encoder_out["src_index_tokens"] = src_index_tokens

        return encoder_out

    def forward(
        self,
        src_tokens: Tensor,
        dict_feats: Optional[Tuple[Tensor, Tensor, Tensor]],
        contextual_embed: Optional[Tensor],
        char_feats: Optional[Tensor],
        src_lengths: Tensor,
        src_subword_begin_indices: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        beam_size: Optional[int] = None,
        src_index_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        encoder_out = self.get_encoder_out(
            src_tokens=src_tokens,
            dict_feats=dict_feats,
            contextual_embed=contextual_embed,
            char_feats=char_feats,
            src_subword_begin_indices=src_subword_begin_indices,
            src_lengths=src_lengths,
            src_index_tokens=src_index_tokens,
        )
        encoder_mask: Optional[Tensor] = None
        if "encoder_mask" in encoder_out:
            encoder_mask = encoder_out["encoder_mask"]

        length_prediction_result: Dict[str, Tensor] = self.get_length_prediction(
            encoder_out=encoder_out,
            encoder_mask=encoder_mask,
            src_lengths=src_lengths,
            target_lengths=target_lengths,
            beam_size=beam_size,
        )
        tgt_tokens = length_prediction_result["target_tokens"]
        length_prob = length_prediction_result["length_probabilities"]
        length_mask = length_prediction_result["length_mask"]
        beam = length_prediction_result["beam"]

        bsz = src_tokens.size(0)
        max_len = tgt_tokens.size(1)

        tiled_encoder_out = self.model.encoder.prepare_for_nar_inference(
            self.length_beam_size, encoder_out
        )

        # OneStep Generation
        tgt_tokens, token_probs, token_logits = self.generate(
            tiled_encoder_out=tiled_encoder_out,
            tgt_tokens=tgt_tokens,
        )
        token_probs = token_probs.view(bsz, self.length_beam_size, max_len).log()
        lprobs = token_probs.sum(-1)
        hypotheses = tgt_tokens.view(bsz, self.length_beam_size, max_len)
        lprobs = lprobs.view(bsz, self.length_beam_size)
        tgt_lengths = (1 - length_mask).sum(-1)

        hyp_score = self.beam_ranking_algorithm(
            token_lprob=lprobs, length_lprob=length_prob, target_lengths=tgt_lengths
        )
        sorted_scores, indices = torch.sort(hyp_score, dim=-1, descending=True)

        all_indices = torch.arange(bsz).unsqueeze(-1)
        hypotheses = hypotheses[all_indices, indices]
        return hypotheses, beam, sorted_scores.exp(), token_probs, token_logits

    def generate(
        self,
        tiled_encoder_out: Dict[str, Tensor],
        tgt_tokens: torch.Tensor,
    ):
        # One step Generation
        pad_mask = tgt_tokens.eq(self.trg_vocab.pad_idx)

        tgt_tokens, token_probs, token_logits = self.generate_non_autoregressive(
            tiled_encoder_out, tgt_tokens
        )
        tgt_tokens[pad_mask] = torch.tensor(
            self.trg_vocab.pad_idx, device=tgt_tokens.device
        ).long()
        token_probs[pad_mask] = torch.tensor(
            1.0, device=token_probs.device, dtype=token_probs.dtype
        )
        return tgt_tokens, token_probs, token_logits

    def get_length_prediction(
        self,
        encoder_out: Dict[str, Tensor],
        encoder_mask: Optional[Tensor],
        src_lengths: Tensor,
        target_lengths: Optional[Tensor] = None,
        beam_size: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        predicted_tgt_length, _ = self.length_prediction_model(
            encoder_out["encoder_out"], encoder_mask
        )
        if beam_size is not None and beam_size != self.length_beam_size:
            self.length_beam_size = beam_size
        if self.clip_target_length:
            beam_vals, beam = predicted_tgt_length.topk(self.length_beam_size, dim=1)
            len_clips = self.get_clip_length(src_lengths)
            len_clips = len_clips.reshape(-1, 1).repeat(1, self.length_beam_size)
            acceptable_lens_mask = torch.le(beam, len_clips)

            beam = beam * acceptable_lens_mask + torch.logical_not(
                acceptable_lens_mask
            ) * torch.ones_like(beam, dtype=beam.dtype, device=beam.device)

            beam_vals = beam_vals * acceptable_lens_mask + torch.logical_not(
                acceptable_lens_mask
            ) * torch.full(
                beam_vals.size(),
                float("-inf"),
                dtype=beam_vals.dtype,
                device=beam_vals.device,
            )
        else:
            beam_vals, beam = predicted_tgt_length.topk(self.length_beam_size, dim=1)

        # make sure no beams are 0 (integration test)
        beam[beam == 0] += 1
        length_prob = torch.gather(predicted_tgt_length, 1, beam)

        if self.use_gold_length:
            assert target_lengths is not None
            beam = target_lengths.reshape(-1, 1)
            self.length_beam_size = 1
            length_prob = torch.ones(beam.size(), device=target_lengths.device)

        tgt_tokens, length_mask = prepare_masked_target_for_lengths(
            beam,
            self.trg_vocab.mask_idx,
            self.trg_vocab.pad_idx,
            self.length_beam_size,
        )

        return {
            "target_tokens": tgt_tokens,
            "length_mask": length_mask,
            "length_probabilities": length_prob,
            "beam": beam,
        }

    def get_clip_length(self, src_lengths: Tensor):
        predicted = (
            torch.tensor(
                self.targetlen_a, dtype=src_lengths.dtype, device=src_lengths.device
            )
            * src_lengths
            * src_lengths
            + torch.tensor(
                self.targetlen_b, dtype=src_lengths.dtype, device=src_lengths.device
            )
            * src_lengths
            + torch.tensor(
                self.targetlen_c, dtype=src_lengths.dtype, device=src_lengths.device
            )
        )
        capped = torch.min(
            predicted,
            torch.tensor(
                self.targetlen_cap, dtype=src_lengths.dtype, device=src_lengths.device
            ),
        )
        return capped

    @torch.jit.export
    def generate_hypo(
        self, tensors: Dict[str, Tensor]
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        Generates hypotheses using beam search, also returning their scores

        Inputs:
            - tensors: dictionary containing needed tensors for generation

        Outputs:
            - (hypos, lens): tuple of Tensors
                - hypos: Tensor of shape (batch_size, beam_size, MAX) containing the generated tokens. MAX refers to the longest sequence in batch.
                - lens: Tensor of shape (batch_size, beam_size) containing generated sequence lengths
            - _hypo_scores: Tensor of shape (batch_size, beam_size) containing the scores for each generated sequence
        """
        actual_src_tokens = tensors["src_tokens"]
        dict_feats: Optional[Tuple[Tensor, Tensor, Tensor]] = None
        contextual_embed: Optional[Tensor] = None
        char_feats: Optional[Tensor] = None

        if "dict_tokens" in tensors:
            dict_feats = (
                tensors["dict_tokens"],
                tensors["dict_weights"],
                tensors["dict_lengths"],
            )

        if "contextual_embed" in tensors:
            contextual_embed = tensors["contextual_embed"]

        if "char_feats" in tensors:
            char_feats = tensors["char_feats"]

        hypos, lens, hypo_scores, _, _ = self.forward(
            actual_src_tokens,
            dict_feats,
            contextual_embed,
            char_feats,
            tensors["src_lengths"],
            src_subword_begin_indices=tensors.get("src_subword_begin_indices"),
            target_lengths=tensors["target_lengths"],
            beam_size=self.length_beam_size,
            src_index_tokens=tensors.get("src_index_tokens"),
        )
        return (hypos, lens), hypo_scores

    def generate_non_autoregressive(self, encoder_out: Dict[str, Tensor], tgt_tokens):
        decoder_out_tuple = self.model.decoder(tgt_tokens, encoder_out)
        tgt_tokens, token_probs, token_logits = self.model.decoder.get_probs(
            decoder_out_tuple
        )
        return tgt_tokens, token_probs, token_logits
