#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from caffe2.python.crf_predict import apply_crf
from pytext.common.constants import Padding
from pytext.utils.cuda import GetTensor
from torch.autograd import Variable


class CRF(nn.Module):
    """
    Compute the log-likelihood of the input assuming a conditional random field
    model.

    Args:
        num_tags: The number of tags
    """

    def __init__(self, num_tags: int) -> None:
        if num_tags <= 0:
            raise ValueError(f"Invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        # Add two states at the end to accommodate start and end states
        # (i,j) element represents the probability of transitioning from state i to j
        self.transitions = nn.Parameter(torch.Tensor(num_tags + 2, num_tags + 2))
        self.start_tag = num_tags
        self.end_tag = num_tags + 1
        self.reset_parameters()
        # TODO Remove hardcoding to read from metadata
        self.ignore_index = Padding.WORD_LABEL_PAD_IDX

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.start_tag] = -10000
        self.transitions.data[self.end_tag, :] = -10000

    def get_transitions(self):
        return self.transitions.data

    def set_transitions(self, transitions: torch.Tensor = None):
        self.transitions.data = transitions

    def forward(
        self,
        emissions: torch.FloatTensor,
        tags: torch.LongTensor,
        ignore_index=Padding.WORD_LABEL_PAD_IDX,
        reduce: bool = True,
    ) -> Variable:
        """
        Compute log-likelihood of input.

        Args:
            emissions: Emission values for different tags for each input. The
                expected shape is batch_size * seq_len * num_labels. Padding is
                should be on the right side of the input.
            tags: Actual tags for each token in the input. Expected shape is
                batch_size * seq_len
        """
        self.ignore_index = ignore_index
        mask = self._make_mask_from_targets(tags)

        numerator = self._compute_joint_llh(emissions, tags, mask)
        denominator = self._compute_log_partition_function(emissions, mask)

        llh = numerator - denominator
        return llh if not reduce else torch.mean(llh)

    def decode(
        self, emissions: torch.FloatTensor, seq_lens: torch.LongTensor
    ) -> torch.Tensor:
        """
        Given a set of emission probabilities, return the predicted tags.

        Args:
            emissions: Emission probabilities with expected shape of
                batch_size * seq_len * num_labels
            seq_lens: Length of each input.
        """
        mask = self._make_mask_from_seq_lens(seq_lens)
        result = self._viterbi_decode(emissions, mask)
        return result

    def _compute_joint_llh(
        self,
        emissions: torch.FloatTensor,
        tags: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        seq_len = emissions.shape[1]

        # Log-likelihood for a given input is calculated by using the known
        # correct tag for each timestep and its respective emission value.
        # Since actual tags for each time step is also known, sum of transition
        # probabilities is also calculated.
        # Sum of emission and transition probabilities gives the final score for
        # the input.
        llh = self.transitions[self.start_tag, tags[:, 0]].unsqueeze(1)
        llh += emissions[:, 0, :].gather(1, tags[:, 0].view(-1, 1)) * mask[
            :, 0
        ].unsqueeze(1)

        for idx in range(1, seq_len):
            old_state, new_state = (
                tags[:, idx - 1].view(-1, 1),
                tags[:, idx].view(-1, 1),
            )
            emission_scores = emissions[:, idx, :].gather(1, new_state)
            transition_scores = self.transitions[old_state, new_state]
            llh += (emission_scores + transition_scores) * mask[:, idx].unsqueeze(1)

        # Index of the last tag is calculated by taking the sum of mask matrix
        # for each input row and subtracting 1 from the sum.
        last_tag_indices = mask.sum(1, dtype=torch.long) - 1
        last_tags = tags.gather(1, last_tag_indices.view(-1, 1))

        llh += self.transitions[last_tags.squeeze(1), self.end_tag].unsqueeze(1)

        return llh.squeeze(1)

    def _compute_log_partition_function(
        self, emissions: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.Tensor:
        seq_len = emissions.shape[1]

        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transitions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(0)
            broadcast_logprob = log_prob.unsqueeze(2)
            score = broadcast_logprob + broadcast_emissions + broadcast_transitions

            score = torch.logsumexp(score, 1)
            log_prob = score * mask[:, idx].unsqueeze(1) + log_prob.squeeze(1) * (
                1.0 - mask[:, idx].unsqueeze(1)
            )

        log_prob += self.transitions[: self.start_tag, self.end_tag].unsqueeze(0)
        return torch.logsumexp(log_prob.squeeze(1), 1)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.Tensor:
        seq_len = emissions.shape[1]
        mask = mask.to(torch.uint8)

        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        # At each step, we need to keep track of the total score, as if this step
        # was the last valid step.
        end_scores = log_prob + self.transitions[
            : self.start_tag, self.end_tag
        ].unsqueeze(0)

        best_scores_list = []
        # If the element has only token, empty tensor in best_paths helps
        # torch.cat() from crashing
        best_paths_list = [GetTensor(torch.Tensor().long())]
        best_scores_list.append(end_scores.unsqueeze(1))

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transmissions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(0)
            broadcast_log_prob = log_prob.unsqueeze(2)

            score = broadcast_emissions + broadcast_transmissions + broadcast_log_prob

            max_scores, max_score_indices = torch.max(score, 1)

            best_paths_list.append(max_score_indices.unsqueeze(1))

            # Storing the scores incase this was the last step.
            end_scores = max_scores + self.transitions[
                : self.start_tag, self.end_tag
            ].unsqueeze(0)

            best_scores_list.append(end_scores.unsqueeze(1))
            log_prob = max_scores

        best_scores = torch.cat(best_scores_list, 1).float()
        best_paths = torch.cat(best_paths_list, 1)

        _, max_indices_from_scores = torch.max(best_scores, 2)

        valid_index_tensor = GetTensor(torch.tensor(0)).long()
        padding_tensor = GetTensor(torch.tensor(self.ignore_index)).long()

        # Label for the last position is always based on the index with max score
        # For illegal timesteps, we set as ignore_index
        labels = max_indices_from_scores[:, seq_len - 1]
        labels = self._mask_tensor(labels, 1.0 - mask[:, seq_len - 1], padding_tensor)

        all_labels = labels.unsqueeze(1).long()

        # For Viterbi decoding, we start at the last position and go towards first
        for idx in range(seq_len - 2, -1, -1):
            # There are two ways to obtain labels for tokens at a particular position.

            # Option 1: Use the labels obtained from the previous position to index
            # the path in present position. This is used for all positions except
            # last position in the sequence.
            # Option 2: Find the indices with maximum scores obtained during
            # viterbi decoding. This is used for the token at the last position

            # For option 1 need to convert invalid indices to 0 so that lookups
            # dont fail.
            indices_for_lookup = all_labels[:, -1].clone()
            indices_for_lookup = self._mask_tensor(
                indices_for_lookup,
                indices_for_lookup == self.ignore_index,
                valid_index_tensor,
            )

            # Option 1 is used here when previous timestep (idx+1) was valid.
            indices_from_prev_pos = (
                best_paths[:, idx, :]
                .gather(1, indices_for_lookup.view(-1, 1).long())
                .squeeze(1)
            )
            indices_from_prev_pos = self._mask_tensor(
                indices_from_prev_pos, (1.0 - mask[:, idx + 1]), padding_tensor
            )

            # Option 2 is used when last timestep was not valid which means idx+1
            # is the last position in the sequence.
            indices_from_max_scores = max_indices_from_scores[:, idx]
            indices_from_max_scores = self._mask_tensor(
                indices_from_max_scores, mask[:, idx + 1], padding_tensor
            )

            # We need to combine results from 1 and 2 as rows in a batch can have
            # sequences of varying lengths
            labels = torch.where(
                indices_from_max_scores == self.ignore_index,
                indices_from_prev_pos,
                indices_from_max_scores,
            )

            # Set to ignore_index if present state is not valid.
            labels = self._mask_tensor(labels, (1 - mask[:, idx]), padding_tensor)
            all_labels = torch.cat((all_labels, labels.view(-1, 1).long()), 1)

        return torch.flip(all_labels, [1])

    def _make_mask_from_targets(self, targets):
        mask = targets.ne(self.ignore_index).float()
        return mask

    def _make_mask_from_seq_lens(self, seq_lens):
        seq_lens = seq_lens.view(-1, 1)
        max_len = torch.max(seq_lens)
        range_tensor = GetTensor(torch.arange(max_len)).unsqueeze(0)
        range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))
        mask = (range_tensor < seq_lens).float()
        return mask

    def _mask_tensor(self, score_tensor, mask_condition, mask_value):
        masked_tensor = torch.where(mask_condition, mask_value, score_tensor)
        return masked_tensor

    def export_to_caffe2(self, workspace, init_net, predict_net, logits_output_name):
        """
        Exports the crf layer to caffe2 by manually adding the necessary operators
        to the init_net and predict net.

        Args:
            init_net: caffe2 init net created by the current graph
            predict_net: caffe2 net created by the current graph
            workspace: caffe2 current workspace
            output_names: current output names of the caffe2 net
            py_model: original pytorch model object

        Returns:
            string: The updated predictions blob name
        """
        crf_transitions = init_net.AddExternalInput(init_net.NextName())
        workspace.FeedBlob(str(crf_transitions), self.get_transitions().numpy())
        logits_squeezed = predict_net.Squeeze(logits_output_name, dims=[0])
        new_logits = apply_crf(
            init_net, predict_net, crf_transitions, logits_squeezed, self.num_tags
        )
        new_logits = predict_net.ExpandDims(new_logits, dims=[0])
        predict_net.Copy(new_logits, logits_output_name)
        return logits_output_name
