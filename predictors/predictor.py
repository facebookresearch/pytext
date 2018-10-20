#!/usr/bin/env python3

from typing import Any, Dict, List

import torch.nn as nn
from pytext.data import DataHandler


class Predictor:
    def __init__(self, model: nn.Module, data_handler: DataHandler) -> None:
        model.eval()
        self.model = model
        self.data_handler = data_handler

    def predict(self, data: List[Dict[str, Any]]) -> List[List[Dict]]:
        input, context = self.data_handler.get_predict_iter(data)
        # Invoke the model
        model_out = self.model(*input)
        return self.fill_predictions(model_out, context)

    def fill_predictions(
        self, model_output: List[Any], context: Dict[str, Any]
    ) -> List[List[Dict]]:
        raise NotImplementedError()
