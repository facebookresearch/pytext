#!/usr/bin/env python3

# Define export input and input names for the models
import torch
from pytext.common.constants import PredictorInputNames


TOKENS_INPUT_NAMES = [PredictorInputNames.TOKENS_IDS, PredictorInputNames.TOKENS_LENS]

DICT_FEAT_INPUT_NAMES = [
    PredictorInputNames.DICT_FEAT_IDS,
    PredictorInputNames.DICT_FEAT_WEIGHTS,
    PredictorInputNames.DICT_FEAT_LENS,
]

TOKENS_INPUT_VALUES = (
    torch.tensor([[1], [1]], dtype=torch.long),
    torch.tensor([1, 1], dtype=torch.long),
)

DICT_FEAT_INPUT_VALUES = (
    torch.tensor([[2], [2]], dtype=torch.long),
    torch.tensor([[1.5], [2.5]], dtype=torch.float),
    torch.tensor([1, 1], dtype=torch.long),
)
