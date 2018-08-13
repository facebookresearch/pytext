#!/usr/bin/env python3
from fblearner_predictor.model.definitions.ttypes import PredictionValue
from pytext.server.util import convert_to_model_predictions
import unittest


class PyTextServiceTest(unittest.TestCase):
    def test_convert_predict_result(self):
        result = [
            [{"name": "a", "value": 1}, {"name": "b", "value": [2, 3]}],
            [{"name": "c", "value": 1.0}, {"name": "d", "value": [2.0, 3.0]}],
            [{"name": "e", "value": ["foo", "bar"]}],
        ]
        pred = convert_to_model_predictions(result)
        self.assertEqual(pred[0].value[0].name, 'a')
        self.assertEqual(pred[0].value[0].value.getType(), PredictionValue.SINGLE_INT)
        self.assertEqual(pred[0].value[1].value.getType(), PredictionValue.I64S)
        self.assertEqual(pred[1].value[0].value.getType(), PredictionValue.SINGLE_FLOAT)
        self.assertEqual(pred[1].value[1].value.getType(), PredictionValue.FLOATS)
        self.assertEqual(pred[2].value[0].value.getType(), PredictionValue.STRINGS)
