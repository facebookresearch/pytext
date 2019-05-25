#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from collections import Counter

import torch
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.config.module_config import PoolingType
from pytext.models.embeddings import DictEmbedding, EmbeddingList, WordEmbedding
from pytext.models.semantic_parsers.rnng.rnng_data_structures import (
    CompositionalNN,
    CompositionalSummationNN,
    Element,
    ParserState,
    StackLSTM,
)
from pytext.models.semantic_parsers.rnng.rnng_parser import RNNGParser
from torchtext.vocab import Vocab


class RNNGDataStructuresTest(unittest.TestCase):
    def test_StackLSTM(self):
        lstm_dim = 100
        lstm_num_layers = 2

        element_root = Element("Root")
        element_node = Element("Node")

        lstm = nn.LSTM(lstm_dim, lstm_dim, num_layers=lstm_num_layers)
        empty_embedding = torch.zeros(1, lstm_dim)
        stackLSTM = StackLSTM(lstm)

        stackLSTM.push(empty_embedding, element_node)
        self.assertEqual(len(stackLSTM), 1)
        self.assertEqual(stackLSTM.element_from_top(0), element_node)
        self.assertEqual(stackLSTM.element_from_top(1), element_root)
        self.assertEqual(stackLSTM.embedding().shape, empty_embedding.shape)

        self.assertEqual(stackLSTM.pop()[1], element_node)
        self.assertEqual(len(stackLSTM), 0)
        self.assertEqual(stackLSTM.element_from_top(0), element_root)
        self.assertTrue(torch.equal(stackLSTM.embedding(), empty_embedding))

    def test_CompositionFunction(self):
        lstm_dim = 100
        embedding = torch.ones(1, lstm_dim)
        input_sequence = [embedding for _ in range(10)]

        compositionalNN = CompositionalNN(lstm_dim)
        self.assertEqual(compositionalNN(input_sequence).shape, embedding.shape)

        compositionalSummationNN = CompositionalSummationNN(lstm_dim)
        self.assertEqual(
            compositionalSummationNN(input_sequence).shape, embedding.shape
        )


class RNNGParserTest(unittest.TestCase):
    def setUp(self):
        actions_counter = Counter()
        for action in [
            "IN:A",
            "IN:B",
            "IN:UNSUPPORTED",
            "REDUCE",
            "SHIFT",
            "SL:C",
            "SL:D",
        ]:
            actions_counter[action] += 1
        actions_vocab = Vocab(actions_counter, specials=[])

        self.parser = RNNGParser(
            ablation=RNNGParser.Config.AblationParams(),
            constraints=RNNGParser.Config.RNNGConstraints(),
            lstm_num_layers=2,
            lstm_dim=20,
            max_open_NT=10,
            dropout=0.2,
            actions_vocab=actions_vocab,
            shift_idx=4,
            reduce_idx=3,
            ignore_subNTs_roots=[2],
            valid_NT_idxs=[0, 1, 2, 5, 6],
            valid_IN_idxs=[0, 1, 2],
            valid_SL_idxs=[5, 6],
            embedding=EmbeddingList(
                embeddings=[
                    WordEmbedding(
                        num_embeddings=5,
                        embedding_dim=20,
                        embeddings_weight=None,
                        init_range=[-1, 1],
                        unk_token_idx=4,
                        mlp_layer_dims=[],
                    ),
                    DictEmbedding(
                        num_embeddings=4, embed_dim=10, pooling_type=PoolingType.MEAN
                    ),
                ],
                concat=True,
            ),
            p_compositional=CompositionalNN(lstm_dim=20),
        )
        self.parser.train()

    def populate_buffer(self):
        state = ParserState(self.parser)
        for _ in range(2):
            state.buffer_stackrnn.push(torch.zeros(1, 30), Element("Token"))
        return state

    def check_valid_actions(self, state, actions):
        self.assertSetEqual(set(self.parser.valid_actions(state)), set(actions))

    def test_valid_actions_unconstrained(self):

        self.parser.constraints_intent_slot_nesting = False
        self.parser.constraints_no_slots_inside_unsupported = False
        state = self.populate_buffer()

        # Valid Actions at beginning: all nonterminals
        self.check_valid_actions(state, self.parser.valid_NT_idxs)

        # After pushing IN:A: all nonterminals, SHIFT
        self.parser.push_action(state, self.parser.actions_vocab.stoi["IN:A"])
        self.check_valid_actions(
            state, self.parser.valid_NT_idxs + [self.parser.shift_idx]
        )

        # After pushing SL:C and SHIFT: all nonterminals, SHIFT, REDUCE
        self.parser.push_action(state, self.parser.actions_vocab.stoi["SL:C"])
        self.parser.push_action(state, self.parser.actions_vocab.stoi["SHIFT"])
        self.check_valid_actions(
            state,
            self.parser.valid_NT_idxs
            + [self.parser.shift_idx]
            + [self.parser.reduce_idx],
        )

        # After all SHIFTs: only REDUCE
        self.parser.push_action(state, self.parser.actions_vocab.stoi["SHIFT"])
        self.check_valid_actions(state, [self.parser.reduce_idx])

        # After all REDUCEs, no valid actions
        self.parser.push_action(state, self.parser.actions_vocab.stoi["REDUCE"])
        self.parser.push_action(state, self.parser.actions_vocab.stoi["REDUCE"])
        self.check_valid_actions(state, [])

    def test_valid_actions_constraint_insl(self):

        self.parser.constraints_intent_slot_nesting = True
        self.parser.constraints_no_slots_inside_unsupported = False
        state = self.populate_buffer()

        # Valid Actions at beginning: all intents
        self.check_valid_actions(state, self.parser.valid_IN_idxs)

        # After pushing IN:A: all slots, SHIFT
        self.parser.push_action(state, self.parser.actions_vocab.stoi["IN:A"])
        self.check_valid_actions(
            state, self.parser.valid_SL_idxs + [self.parser.shift_idx]
        )

        # After pushing SL:C: all intents, SHIFT
        self.parser.push_action(state, self.parser.actions_vocab.stoi["SL:C"])
        self.check_valid_actions(
            state, self.parser.valid_IN_idxs + [self.parser.shift_idx]
        )

        # After all SHIFTs: only REDUCE
        self.parser.push_action(state, self.parser.actions_vocab.stoi["SHIFT"])
        self.parser.push_action(state, self.parser.actions_vocab.stoi["SHIFT"])
        self.check_valid_actions(state, [self.parser.reduce_idx])

        # After all REDUCEs, no valid actions
        self.parser.push_action(state, self.parser.actions_vocab.stoi["REDUCE"])
        self.parser.push_action(state, self.parser.actions_vocab.stoi["REDUCE"])
        self.check_valid_actions(state, [])

    def test_valid_actions_constraint_unsupported(self):

        self.parser.constraints_intent_slot_nesting = True
        self.parser.constraints_no_slots_inside_unsupported = True
        state = self.populate_buffer()

        # Valid Actions at beginning: all intents
        self.check_valid_actions(state, self.parser.valid_IN_idxs)

        # Needed to make test logic work
        state.predicted_actions_idx.append(
            self.parser.actions_vocab.stoi["IN:UNSUPPORTED"]
        )
        # After pushing IN:UNSUPPORTED: SHIFT
        self.parser.push_action(state, self.parser.actions_vocab.stoi["IN:UNSUPPORTED"])
        self.check_valid_actions(state, [self.parser.shift_idx])

    def test_forward_shapes(self):
        self.parser.eval(Stage.EVAL)
        tokens = torch.tensor([[0, 1, 2, 3]])
        seq_lens = torch.tensor([tokens.shape[1]])
        dict_feat = (
            torch.tensor([[1, 1, 1, 1, 1, 1, 3, 1]]),
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            torch.tensor([[1, 1, 2, 1]]),
        )

        actions, scores = self.parser(
            tokens=tokens, seq_lens=seq_lens, dict_feat=dict_feat
        )[0]
        self.assertGreater(actions.shape[1], tokens.shape[1])
        self.assertEqual(actions.shape[0:2], scores.shape[0:2])
        self.assertEqual(scores.shape[2], len(self.parser.actions_vocab.itos))

        # Beam Search Test
        self.parser.eval(Stage.TEST)
        results = self.parser(
            tokens=tokens, seq_lens=seq_lens, dict_feat=dict_feat, beam_size=3, top_k=3
        )
        self.assertEqual(len(results), 3)
        for actions, scores in results:
            self.assertGreater(actions.shape[1], tokens.shape[1])
            self.assertEqual(actions.shape[0:2], scores.shape[0:2])
            self.assertEqual(scores.shape[2], len(self.parser.actions_vocab.itos))
