#!/usr/bin/env python3

import torch
from .rnng_pytext_predictor import RNNGPyTextPredictor
from .seq2seq_pytext_predictor import SEQ2SEQPyTextPredictor


def load_predictor(load_path: str) -> None:

    predictor_state = torch.load(load_path, map_location=lambda storage, loc: storage)
    # TODO should use the common pytext way to load predictor
    # Handle compositional models
    if predictor_state.get("compositional", False):
        return RNNGPyTextPredictor(load_path)

    if predictor_state.get("fairseq", False):
        return SEQ2SEQPyTextPredictor(predictor_state)

    raise Exception("this function is deprecated, please use load in serialize.py")
