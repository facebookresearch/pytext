#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given
from pytext.common.constants import Padding
from pytext.models.crf import CRF
from scipy.special import logsumexp


class CRFTest(hu.HypothesisTestCase):
    @given(
        num_tags=st.integers(2, 10),
        seq_lens=st.lists(
            elements=st.integers(min_value=1, max_value=10), min_size=1, max_size=10
        ),
    )
    def test_crf_forward(self, num_tags, seq_lens):

        crf_model = CRF(
            num_tags,
            ignore_index=Padding.WORD_LABEL_PAD_IDX,
            default_label_pad_index=Padding.DEFAULT_LABEL_PAD_IDX,
        )

        total_manual_loss = 0

        padded_inputs = []
        padded_targets = []

        max_num_words = max(seq_lens)

        for seq_len in seq_lens:

            target_tokens = np.random.randint(1, num_tags, size=(1, seq_len))
            padded_targets.append(
                np.concatenate(
                    [target_tokens, np.zeros((1, max_num_words - seq_len))], axis=1
                )
            )

            input_emission = np.random.rand(seq_len, num_tags)
            padded_inputs.append(
                np.concatenate(
                    [input_emission, np.zeros((max_num_words - seq_len, num_tags))],
                    axis=0,
                )
            )

            manual_loss = self._compute_loss_manual(
                input_emission,
                num_tags,
                target_tokens.reshape(-1),
                crf_model.get_transitions().tolist(),
            )
            crf_loss = crf_model(
                torch.tensor(input_emission, dtype=torch.float).unsqueeze(0),
                torch.tensor(target_tokens),
            )

            # Loss returned by CRF model for each input should be equal to
            # manually calculated loss
            self.assertAlmostEqual(manual_loss, -1 * crf_loss.item(), places=4)
            total_manual_loss += manual_loss

        # Loss returned by CRF model for batched input should be equal to
        # average of manually calculated loss
        batched_crf_loss = crf_model(
            torch.tensor(padded_inputs, dtype=torch.float),
            torch.tensor(padded_targets, dtype=torch.long).squeeze(1),
        )
        self.assertAlmostEqual(
            total_manual_loss / len(seq_lens), -1 * batched_crf_loss.item(), places=4
        )

    def _compute_loss_manual(self, predictions, num_tags, labels, transitions):
        low_score = -1000
        b_s = np.array([[low_score] * num_tags + [0, low_score]]).astype(np.float32)
        e_s = np.array([[low_score] * num_tags + [low_score, 0]]).astype(np.float32)
        predictions = np.concatenate(
            [predictions, low_score * np.ones((predictions.shape[0], 2))], axis=1
        )
        predictions = np.concatenate([b_s, predictions, e_s], axis=0)
        b_id = np.array([num_tags], dtype=np.int32)
        e_id = np.array([num_tags + 1], dtype=np.int32)
        labels = np.concatenate([b_id, labels, e_id], axis=0)
        curr_state = predictions[0]
        input_states = predictions[1:]

        for input_state in input_states:
            prev = np.expand_dims(curr_state, axis=1)
            curr_input = np.expand_dims(input_state, axis=0)
            curr_state = logsumexp(prev + curr_input + transitions, axis=0)

        total_score = logsumexp(curr_state, axis=0)
        # Compute best path score
        unary_scores = sum(w[labels[i]] for i, w in enumerate(predictions))
        binary_scores = sum(transitions[a][b] for a, b in zip(labels[:-1], labels[1:]))
        loss = total_score - (binary_scores + unary_scores)
        return loss

    @given(
        num_tags=st.integers(2, 10),
        seq_lens=st.lists(
            elements=st.integers(min_value=1, max_value=10), min_size=1, max_size=10
        ),
    )
    def test_crf_decode_torchscript(self, num_tags, seq_lens):
        crf_model = CRF(
            num_tags,
            ignore_index=Padding.WORD_LABEL_PAD_IDX,
            default_label_pad_index=Padding.DEFAULT_LABEL_PAD_IDX,
        )
        crf_model.eval()
        scripted_crf_model = torch.jit.script(crf_model)

        max_num_words = max(seq_lens)
        padded_inputs = []
        for seq_len in seq_lens:
            input_emission = np.random.rand(seq_len, num_tags)
            padded_inputs.append(
                np.concatenate(
                    [input_emission, np.zeros((max_num_words - seq_len, num_tags))],
                    axis=0,
                )
            )
            crf_decode = crf_model.decode(
                torch.tensor(input_emission, dtype=torch.float).unsqueeze(0),
                torch.tensor([seq_len]),
            )

            scripted_crf_decode = scripted_crf_model.decode(
                torch.tensor(input_emission, dtype=torch.float).unsqueeze(0),
                torch.tensor([seq_len]),
            )

            self.assertTrue(torch.allclose(crf_decode, scripted_crf_decode))

        batched_emissions = torch.tensor(padded_inputs, dtype=torch.float)
        batched_seq_lens = torch.tensor(seq_lens)
        crf_batch_decode = crf_model.decode(batched_emissions, batched_seq_lens)
        scriped_crf_batch_decode = scripted_crf_model.decode(
            batched_emissions, batched_seq_lens
        )
        self.assertTrue(torch.allclose(crf_batch_decode, scriped_crf_batch_decode))
