#!/usr/bin/env python3
from fblearner_predictor.model.definitions.ttypes import (
    NamePredictionPair,
    ModelPrediction,
    PredictionValue,
)
from typing import Dict, List, Any


def convert_to_model_predictions(input: List[List[Dict]]) -> List[ModelPrediction]:
    return [
        ModelPrediction(
            named_predictions=[
                convert_to_named_pred_pair(pred_pair) for pred_pair in pred_pairs
            ]
        )
        for pred_pairs in input
    ]


def convert_to_named_pred_pair(pred_pair: Dict) -> NamePredictionPair:
    if "name" not in pred_pair or "value" not in pred_pair:
        raise Exception("invalid pred_pair")
    return NamePredictionPair(
        name=pred_pair["name"], value=convert_to_pred_value(pred_pair["value"])
    )


def convert_to_pred_value(input: Any) -> PredictionValue:
    input_type = type(input)
    if input_type is int:
        return PredictionValue(single_int=input)
    if input_type is float:
        return PredictionValue(single_float=input)
    if input_type is list:
        input_type = type(input[0])
        if input_type is int:
            return PredictionValue(i64s=input)
        if input_type is float:
            return PredictionValue(floats=input)
        if input_type is str:
            return PredictionValue(strings=input)
    raise Exception("unknown input type {}".format(input_type))
