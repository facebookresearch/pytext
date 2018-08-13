#!/usr/bin/env python3

from pytext.common.constants import DatasetFieldName
from typing import List, Dict
import pytext.rnng.predict_parser as predict_parser
import torch
import pandas as pd


class RNNGPyTextPredictor:
    def __init__(self, model_snapshot_path: str) -> None:
        self.predictor = predict_parser.load_model(model_snapshot_path)

    def predict(self, df: pd.DataFrame) -> List[List[Dict]]:
        fields = [DatasetFieldName.TEXT_FIELD]
        if DatasetFieldName.DICT_FIELD in df:
            fields.append(DatasetFieldName.DICT_FIELD)

        raw_inputs = df[fields].values.tolist()

        torch.manual_seed(0)
        raw_predictions = [
            predict_parser.predict_actions_and_tree(
                self.predictor,
                raw_input,
                add_dict_feat=self.predictor.dictfeat_bidict.size() > 0,
            )
            for raw_input in raw_inputs
        ]

        predictions = []
        prediction_pairs = []

        for raw_prediction in raw_predictions:
            pred_list, pred_tree, pred_scores = raw_prediction
            actions, tokens = map(list, zip(*pred_list))
            prediction_pairs.append({"name": "actions", "value": actions})
            prediction_pairs.append({"name": "tokens", "value": tokens})
            prediction_pairs.append({"name": "scores", "value": pred_scores})
            prediction_pairs.append(
                {"name": "pretty_print", "value": [pred_tree.flat_str()]}
            )
            predictions.append(prediction_pairs)

        return predictions
