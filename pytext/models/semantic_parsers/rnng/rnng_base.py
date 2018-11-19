#!/usr/bin/env python3

from enum import Enum
from typing import List, Tuple

import numpy as np
import pytext.utils.cuda_utils as cuda_utils
import torch as torch
import torch.nn as nn
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.data import CommonMetadata
from pytext.models.module import create_module
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.semantic_parsers.rnng.rnng_bindings import Parser as RNNGParserCpp
from pytext.models.semantic_parsers.rnng.rnng_data_structures import (
    CompositionalNN,
    CompositionalSummationNN,
)
from pytext.models.semantic_parsers.rnng.rnng_py import RNNGParserPy


class CompositionalType(Enum):
    BLSTM = "blstm"
    SUM = "sum"


class AblationParams(ConfigBase):
    use_buffer: bool = True
    use_stack: bool = True
    use_action: bool = True
    use_last_open_NT_feature: bool = False


class RNNGConstraints(ConfigBase):
    intent_slot_nesting: bool = True
    ignore_loss_for_unsupported: bool = False
    no_slots_inside_unsupported: bool = True


class RNNGParser(nn.Module, Component):
    __COMPONENT_TYPE__ = ComponentType.MODEL

    class Config(ConfigBase):
        version: int = 0
        lstm: BiLSTM.Config = BiLSTM.Config()
        ablation: AblationParams = AblationParams()
        constraints: RNNGConstraints = RNNGConstraints()
        max_open_NT: int = 10
        dropout: float = 0.1
        compositional_type: CompositionalType = CompositionalType.BLSTM
        use_cpp: bool = False

    @classmethod
    def from_config(cls, model_config, feature_config, metadata: CommonMetadata):
        if model_config.use_cpp:
            model = RNNGParserCpp(
                RNNGParser.get_cpp_model_config_list(model_config, feature_config),
                metadata.actions_vocab.itos,
                metadata.features[DatasetFieldName.TEXT_FIELD].vocab.itos,
                metadata.features[DatasetFieldName.DICT_FIELD].vocab.itos,
            )
            pretrained_embeds_weight = metadata.features[
                DatasetFieldName.TEXT_FIELD
            ].pretrained_embeds_weight
            if pretrained_embeds_weight is not None:
                print("Initializing word embeddings for RNNG CPP model")
                model.init_word_weights(pretrained_embeds_weight)
        else:
            if model_config.compositional_type == CompositionalType.SUM:
                p_compositional = CompositionalSummationNN(
                    lstm_dim=model_config.lstm.lstm_dim
                )
            elif model_config.compositional_type == CompositionalType.BLSTM:
                p_compositional = CompositionalNN(lstm_dim=model_config.lstm.lstm_dim)
            else:
                raise ValueError(
                    "Cannot understand compositional flag {}".format(
                        model_config.compositional_type
                    )
                )

            model = RNNGParserPy(
                ablation=model_config.ablation,
                constraints=model_config.constraints,
                lstm_num_layers=model_config.lstm.num_layers,
                lstm_dim=model_config.lstm.lstm_dim,
                max_open_NT=model_config.max_open_NT,
                dropout=model_config.dropout,
                actions_vocab=metadata.actions_vocab,
                shift_idx=metadata.shift_idx,
                reduce_idx=metadata.reduce_idx,
                ignore_subNTs_roots=metadata.ignore_subNTs_roots,
                valid_NT_idxs=metadata.valid_NT_idxs,
                valid_IN_idxs=metadata.valid_IN_idxs,
                valid_SL_idxs=metadata.valid_SL_idxs,
                embedding=create_module(feature_config, metadata=metadata),
                p_compositional=p_compositional,
            )
        return cls(model)

    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, *inputs):
        if self.training:
            self.model.train()
        else:
            self.model.eval()
        return self.model.forward(*inputs)

    def get_model_params_for_optimizer(self):
        # First arg is None because there aren't params that need sparse gradients.
        return None, self.model.parameters()

    def get_loss(
        self, logits: torch.Tensor, target_actions: torch.Tensor, context: torch.Tensor
    ):
        # action scores is a 2D Tensor of dims sequence_length x number_of_actions
        # targets is a 1D list of integers of length sequence_length

        # Get rid of the batch dimension
        action_scores = logits[1].squeeze(0)
        target_actions = target_actions.squeeze(0)

        action_scores_list = torch.chunk(action_scores, action_scores.size()[0])
        target_vars = [
            cuda_utils.Variable(torch.LongTensor([t])) for t in target_actions
        ]
        losses = [
            self.loss_func(action, target).view(1)
            for action, target in zip(action_scores_list, target_vars)
        ]
        total_loss = torch.sum(torch.cat(losses)) if len(losses) > 0 else None
        return total_loss

    def get_pred(self, logits: Tuple[torch.Tensor, torch.Tensor], *args):
        predicted_action_idx, predicted_action_scores = logits
        predicted_scores = [
            np.exp(np.max(action_scores)).item() / np.sum(np.exp(action_scores)).item()
            for action_scores in predicted_action_scores.detach().squeeze(0).tolist()
        ]

        return predicted_action_idx.tolist(), [predicted_scores]

    def save_modules(self, *args, **kwargs):
        pass

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, model_state):
        self.model.load_state_dict(model_state)

    def parameters(self):
        return self.model.parameters()

    @staticmethod
    def get_cpp_model_config_list(model_config, feature_config) -> List[float]:
        model_config = [
            model_config.version,
            model_config.lstm.lstm_dim,
            model_config.lstm.num_layers,
            feature_config.word_feat.embed_dim,
            model_config.max_open_NT,
            feature_config.dict_feat.embed_dim if feature_config.dict_feat else 0,
            model_config.dropout,
        ]
        return [float(mc) for mc in model_config]
