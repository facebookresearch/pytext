#!/usr/bin/env python3
# Adopted from the open-source AllenNLP implementation of CRF on github
# https://github.com/allenai/allennlp/blob/master/
# allennlp/modules/conditional_random_field.py

from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from caffe2.python.crf_predict import apply_crf
from pytext.common.constants import Padding
from pytext.utils.cuda_utils import Variable as Var
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence


class CRF(nn.Module):
    """
    num_tags : int
        Number of tags passed to ``__init__``.
    transitions : class:`~torch.Tensor` optional to initialize the crf
        Transition score tensor of size ``(num_tags, num_tags)``.
    ----------

    """

    def __init__(self, num_tags: int) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        # add two states at the end to accommadate start and end states
        # (i,j) element represents the prob of transitioning from state i to j
        self.transitions = nn.Parameter(torch.Tensor(num_tags + 2, num_tags + 2))
        self.start_tag = num_tags
        self.end_tag = num_tags + 1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        Also enforce the constraint that we never transfer
        to the start tag not transfer from the stop tag
        """
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.start_tag] = -10000
        self.transitions.data[self.end_tag, :] = -10000

    def get_transitions(self):
        return self.transitions.data

    def set_transitions(self, transitions: torch.Tensor = None):
        self.transitions.data = transitions

    def forward(
        self,
        emissions: Variable,
        tags: Variable,
        ignore_index=Padding.WORD_LABEL_PAD_IDX,
        reduce: bool = True,
    ) -> Variable:
        """Compute the log likelihood of the given sequence of tags and
        emission score.
        ---------
        Inputs:
        emissions : :class:`~torch.autograd.Variable`
            Emission score tensor of size (seq_length, batch_size, num_tags).
        tags : :class:`~torch.autograd.Variable`
            Sequence of tags as ``LongTensor`` of size (seq_length, batch_size).
        reduce : bool
            Whether to sum the log likelihood over the batch.
        -------
        Returns:
        :class:`~torch.autograd.Variable`
            The log likelihood. This will have size (1,) if ``summed=True``,
            ``(batch_size,)`` otherwise.
        """
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if tags.dim() != 2:
            raise ValueError(f"tags must have dimension of 2, got {tags.dim()}")
        if emissions.size()[:2] != tags.size():
            raise ValueError(
                "the first two dimensions of emissions and tags must match, "
                f"got {tuple(emissions.size()[:2])} and {tuple(tags.size())}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )
        self.ignore_index = ignore_index
        mask = self.make_mask_from_targets(tags)

        if tags.size() != mask.size():
            raise ValueError(
                f"size of tags and mask must match, {tuple(tags.size())} "
                f"and {tuple(mask.size())}"
            )

        numerator = self._compute_joint_llh(emissions, tags, mask)
        denominator = self._compute_log_partition_function(emissions, mask)
        llh = numerator - denominator
        return llh if not reduce else torch.mean(llh)

    def make_mask_from_targets(self, targets):
        mask = Var(torch.ByteTensor(targets.size()))
        for i in range(mask.size()[0]):
            mask[i, :] = torch.from_numpy(
                np.asarray(
                    [
                        0 if target.item() == Padding.WORD_LABEL_PAD_IDX else 1
                        for target in targets[i, :]
                    ]
                )
            )
        return mask.contiguous()

    def decode(
        self,
        emissions: Union[Variable, torch.FloatTensor],
        seq_lens: Union[Variable, torch.LongTensor],
    ) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Arguments
        ---------
        emissions: `~torch.autograd.Variable` or :class:`~torch.FloatTensor`
            Emission score tensor of size (batch_size, seq_length, num_tags).
        seq_lens : `~torch.autograd.Variable` or `torch.LongTensor`
            Actual seq length (without padding) tensor of size ``(batch_size,
            seq_length)``.
        Returns
        -------
        List of list containing the best tag sequence for each batch.
        """
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )
        if emissions.size()[:1] != seq_lens.size():
            raise ValueError(
                "batch size of emissions and seq lens must match, "
                f"got {tuple(emissions.size()[:1])} and {tuple(seq_lens.size())}"
            )

        if isinstance(emissions, Variable):
            emissions = emissions.data
        elif isinstance(seq_lens, Variable):
            seq_lens = seq_lens.data

        best_tags = []
        for emission, seq_len in zip(emissions, seq_lens):
            best_tags.append(self._viterbi_decode(emission[:seq_len]))
        # TODO read padding token idx from metadata
        res = pad_sequence(
            best_tags, padding_value=Padding.WORD_LABEL_PAD_IDX, batch_first=True
        )
        return res

    def _compute_joint_llh(
        self, emissions: Variable, tags: Variable, mask: Variable
    ) -> Variable:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.size()[:2] == tags.size()
        assert emissions.size(2) == self.num_tags
        assert mask.size() == tags.size()
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()

        # Start transition score
        llh = self.transitions[self.start_tag, tags[0]]  # (batch_size,)
        for i in range(seq_length - 1):
            cur_tag, next_tag = tags[i], tags[i + 1]
            # Emission score for current tag
            llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]
            # Transition score to next tag
            transition_score = self.transitions[cur_tag, next_tag]
            # Only add transition score if the next tag is not masked (mask == 1)
            llh += transition_score * mask[i + 1]
        # Find last tag index
        last_tag_indices = mask.long().sum(0) - 1  # (batch_size,)
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)

        # End transition score
        llh += self.transitions[last_tags, self.end_tag]
        # Emission score for the last tag, if mask is valid (mask == 1)
        llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]

        return llh

    def _compute_log_partition_function(
        self, emissions: Variable, mask: Variable
    ) -> Variable:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()

        # Start transition score and first emission
        log_prob = (
            self.transitions[self.start_tag, : self.start_tag].view(1, -1)
            + emissions[0]
        )
        # Here, log_prob has size (batch_size, num_tags) where for each batch,
        # j-th column is the log probability that the current timestep has tag j

        for i in range(1, seq_length):
            # Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob.unsqueeze(2)  # (batch_size, num_tags, 1)
            # Broadcast transition score over all instances in the batch
            broadcast_transitions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(
                0
            )  # (1, num_tags, num_tags)
            # Broadcast emission score over all possible current tags
            broadcast_emissions = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)
            # Sum current log probability, transition, and emission scores
            score = broadcast_log_prob + broadcast_transitions + broadcast_emissions
            # (batch_size, num_tags, num_tags)
            # Sum over all possible current tags, but we're in log prob space,
            # so a sum becomes a log-sum-exp
            score = self._log_sum_exp(score, 1)  # (batch_size, num_tags)
            # Set log_prob to the score if this timestep is valid (mask == 1),
            # otherwise leave it alone
            log_prob = score * mask[i].unsqueeze(1) + log_prob * (
                1. - mask[i]
            ).unsqueeze(1)

        # End transition score
        log_prob += self.transitions[: self.start_tag, self.end_tag].view(1, -1)
        # Sum (log-sum-exp) over all possible tags
        return self._log_sum_exp(log_prob, 1)  # (batch_size,)

    def _viterbi_decode(self, emission: torch.FloatTensor) -> List[int]:
        # emission: (seq_length, num_tags)
        assert emission.size(1) == self.num_tags

        seq_length = emission.size(0)

        # Start transition
        viterbi_score = (
            self.transitions[self.start_tag, : self.start_tag].data + emission[0]
        )
        viterbi_path = []
        # Here, viterbi_score has shape of (num_tags,) where value at index i
        # is the score of the best tag sequence so far that ends with tag i
        # viterbi_path saves where the best tags candidate transitioned from;
        # this is used when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best
        # tag sequence for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            broadcast_score = viterbi_score.view(-1, 1)
            # Broadcast emission score for every possible current tag
            broadcast_emission = emission[i].view(1, -1)
            # Compute the score matrix of shape (num_tags, num_tags) where
            # each entry at  row i and column j stores the score of
            # transitioning from tag i to tag j and emitting
            score = (
                broadcast_score
                + self.transitions[: self.start_tag, : self.start_tag].data
                + broadcast_emission
            )
            # Find the maximum score over all possible current tag
            best_score, best_path = score.max(0)  # (num_tags,)
            # Save the score and the path
            viterbi_score = best_score
            viterbi_path.append(best_path)

        # End transition
        viterbi_score += self.transitions[: self.start_tag, self.end_tag].data

        # Find the tag which maximizes the score at the last timestep
        _, best_last_tag = viterbi_score.max(0)
        best_tags = [best_last_tag.item()]

        # We trace back where the best last tag comes from, append that to our
        # best tag sequence, and trace it back again, and so on
        for path in reversed(viterbi_path):
            best_last_tag = path[best_tags[-1]]
            best_tags.append(best_last_tag.item())

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        return torch.LongTensor(best_tags)

    """Exports the crf layer to caffe2 by manually adding the necessary operators
        to the init_net and predict net.
        init_net: caffe2 init net created by the current graph
        predict_net: caffe2 net created by the current graph
        workspace: caffe2 current workspace
        output_names: current output names of the caffe2 net
        py_model: original pytorch model object
        Returns:
        The updated predictions blob name
    """

    def export_to_caffe2(self, workspace, init_net, predict_net, logits_output_name):
        crf_transitions = init_net.AddExternalInput(init_net.NextName())
        workspace.FeedBlob(str(crf_transitions), self.get_transitions().numpy())
        logits_squeezed = predict_net.Squeeze(logits_output_name, dims=[0])
        new_logits = apply_crf(
            init_net, predict_net, crf_transitions, logits_squeezed, self.num_tags
        )
        new_logits = predict_net.ExpandDims(new_logits, dims=[0])
        predict_net.Copy(new_logits, logits_output_name)
        return logits_output_name

    @staticmethod
    def _log_sum_exp(tensor: Variable, dim: int) -> Variable:
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(
            torch.sum(torch.exp(tensor - broadcast_offset), dim)
        )
        # Add offset back
        return offset + safe_log_sum_exp
