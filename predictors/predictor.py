#!/usr/bin/env python3

from pytext.data.data_handler import DataHandler
from typing import List, Any, Dict
import pandas as pd
import torch.nn as nn


class Predictor:
    def __init__(self, model: nn.Module, data_handler: DataHandler) -> None:
        model.eval()
        self.model = model
        self.data_handler = data_handler

    def predict(self, df: pd.DataFrame) -> List[List[Dict]]:
        input, context = self.data_handler.get_predict_batch(df)
        # Invoke the model
        model_out = self.model(*input)
        return self.fill_predictions(model_out, context)

    def fill_predictions(
        self, model_output: List[Any], context: Dict[str, Any]
    ) -> List[List[Dict]]:
        raise NotImplementedError()
