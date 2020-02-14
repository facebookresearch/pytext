#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Tuple

import torch
from torch import Tensor


class BeamDecode(torch.nn.Module):
    """
    Decodes the output of Beam Search to get the top hypotheses
    """

    def __init__(self, eos_token_id, length_penalty, nbest, beam_size, stop_at_eos):
        super().__init__()
        self.eos_token_id: int = eos_token_id
        self.length_penalty: float = length_penalty
        self.nbest: int = nbest
        self.beam_size: int = beam_size
        self.stop_at_eos: bool = stop_at_eos

    @torch.no_grad()
    def forward(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        token_weights: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> List[Tuple[Tensor, float, List[float], Tensor, Tensor]]:

        self._check_dimensions(
            beam_tokens, beam_scores, token_weights, beam_prev_indices, num_steps
        )

        end_states = self._get_all_end_states(
            beam_tokens, beam_scores, beam_prev_indices, num_steps
        )

        # outputs is list of the following for each hypothesis:
        # Tuple[Hypothesis, Hypothesis score, Token level scores,
        #       Attention Weights, Best indices]
        outputs = torch.jit.annotate(
            List[Tuple[Tensor, float, List[float], Tensor, Tensor]], []
        )

        for state_idx in range(len(end_states)):
            state = end_states[state_idx]
            hypothesis_score = float(state[0])
            beam_indices = self._get_output_steps_to_beam_indices(
                state, beam_prev_indices
            )
            beam_output = torch.jit.annotate(List[Tensor], [])
            token_level_scores = torch.jit.annotate(List[float], [])
            position = int(state[1])
            hyp_index = int(state[2])

            # best_indices represents the ending position of one hypothesis,
            # the first index corresponds num_step, the second corresponds beam_index
            best_indices = torch.tensor([position, hyp_index])
            back_alignment_weights = []

            assert position + 1 == len(beam_indices)
            pos = 1
            prev_beam_index = -1
            while pos < len(beam_indices):
                beam_index = beam_indices[pos]
                beam_output.append(beam_tokens[pos][beam_index])
                if pos == 1:
                    # beam_scores[0][:] are all 0s
                    token_level_scores.append(float(beam_scores[pos][beam_index]))
                else:
                    token_level_scores.append(
                        float(beam_scores[pos][beam_index])
                        - float(beam_scores[pos - 1][prev_beam_index])
                    )
                back_alignment_weights.append(token_weights[pos][beam_index].detach())
                prev_beam_index = beam_index
                pos += 1
            outputs.append(
                (
                    torch.stack(beam_output),
                    hypothesis_score,
                    token_level_scores,
                    torch.stack(back_alignment_weights, dim=1),
                    best_indices,
                )
            )

        return outputs

    def _get_output_steps_to_beam_indices(
        self, end_state: Tensor, beam_prev_indices: Tensor
    ) -> List[int]:
        """
        Returns a mapping from each output position and the beam index that was
        picked from the beam search results.
        """
        present_position = int(end_state[1])
        beam_index = int(end_state[2])
        beam_indices = torch.jit.annotate(List[int], [])
        while present_position >= 0:
            beam_indices.insert(0, beam_index)
            beam_index = beam_prev_indices[present_position][beam_index]
            present_position = present_position - 1
        return beam_indices

    def _add_to_end_states(
        self, end_states: List[Tensor], min_score: float, state: Tensor, min_index: int
    ) -> Tuple[List[Tensor], float, int]:
        """
        Maintains a list of atmost `nbest` highest end states
        """
        if len(end_states) < self.nbest:
            end_states.append(state)
            # keep min_score and min_index updated
            if state[0] <= min_score:
                min_score = state[0]
                min_index = len(end_states) - 1
        elif state[0] > min_score:
            # replace worst hypo with the new one
            end_states[min_index] = state
            # find new worst hypo, keep min_score and min_index updated
            min_index = -1
            # not using float("inf") temporarily bc of TorchScript bug
            # using max representable value in fp16
            min_score = 65504.0
            for idx in range(len(end_states)):
                s = end_states[idx]
                if s[0] <= min_score:
                    min_index = idx
                    min_score = s[0]
        return end_states, min_score, min_index

    def _get_all_end_states(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> Tensor:
        """
        Return all end states and hypothesis scores for those end states.
        """
        # not using float("inf") temporarily bc of TorchScript bug
        # using max representable value in fp16
        min_score = 65504.0
        min_index = -1
        end_states = torch.jit.annotate(List[Tensor], [])
        prev_hypo_is_finished = torch.zeros(self.beam_size).byte()

        position = 1
        while position <= num_steps:
            hypo_is_finished = torch.zeros(self.beam_size, dtype=torch.bool)

            for hyp_index in range(self.beam_size):
                prev_pos = beam_prev_indices[position][hyp_index]
                hypo_is_finished[hyp_index] = prev_hypo_is_finished[prev_pos]

                # If hypothesis was completed in the previous index,
                # then just continue
                if hypo_is_finished[hyp_index] == 0:
                    # If the present token is EOS or we have reached max_length
                    # then hypothesis is complete
                    if (beam_tokens[position][hyp_index] == self.eos_token_id) or (
                        position == num_steps
                    ):

                        if self.stop_at_eos:
                            hypo_is_finished[hyp_index] = 1

                        hypo_score = float(beam_scores[position][hyp_index])
                        if self.length_penalty != 0:
                            hypo_score = hypo_score / (position ** self.length_penalty)

                        end_states, min_score, min_index = self._add_to_end_states(
                            end_states,
                            min_score,
                            torch.tensor(
                                [hypo_score, float(position), float(hyp_index)]
                            ),
                            min_index,
                        )

            prev_hypo_is_finished = hypo_is_finished
            position = position + 1

        end_states = torch.stack(end_states)

        _, sorted_end_state_indices = end_states[:, 0].sort(dim=0, descending=True)
        end_states = end_states[sorted_end_state_indices, :]
        return end_states

    def _check_dimensions(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        token_weights: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> None:

        assert (
            beam_tokens.size(1) == self.beam_size
        ), "Dimension of beam_tokens : {} and beam size : {} are not consistent".format(
            beam_tokens.size(), self.beam_size
        )
        assert beam_scores.size(1) == self.beam_size, (
            "Dimension of beam_scores : {} and beam size : {} "
            "are not consistent".format(beam_scores.size(), self.beam_size)
        )
        assert token_weights.size(1) == self.beam_size, (
            "Dimension of token_weights : {} and beam size : {} "
            "are not consistent".format(token_weights.size(), self.beam_size)
        )
        assert (
            beam_prev_indices.size(1) == self.beam_size
        ), "Dimension of beam_prev_indices : {} and beam size : {} "
        "are not consistent".format(beam_prev_indices.size(), self.beam_size)

        assert beam_tokens.size(0) <= num_steps + 1, (
            "Dimension of beam_tokens : {} and num_steps : {} "
            "are not consistent".format(beam_tokens.size(), num_steps)
        )
        assert beam_scores.size(0) <= num_steps + 1, (
            "Dimension of beam_scores : {} and num_steps : {} "
            "are not consistent".format(beam_scores.size(), num_steps)
        )
        assert token_weights.size(0) <= num_steps + 1, (
            "Dimension of token_weights : {} and num_steps : {} "
            "are not consistent".format(token_weights.size(), num_steps)
        )
        assert beam_prev_indices.size(0) <= num_steps + 1, (
            "Dimension of beam_prev_indices : {} and num_steps : {} "
            "are not consistent".format(beam_prev_indices.size(), num_steps)
        )
