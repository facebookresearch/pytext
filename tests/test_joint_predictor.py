#!/usr/bin/env python3

import unittest

import numpy as np
import pandas as pd
from pytext.common.registry import JOB_SPEC, Registry
from pytext.config import PyTextConfig
from pytext.jobspec import *
from pytext.predictors.joint_predictor import JointPredictor
from pytext.serialize import load


class JointPredictorTest(unittest.TestCase):
    def test_predict(self):
        PyTextConfig._field_types["jobspec"].__args__ = tuple(Registry.values(JOB_SPEC))
        model_path = "pytext/tests/data/joint_model.pt"
        test_text = "I want to hear Beethoven's 7th Symphony. Bernstein only."
        test_dict = ""
        _, model, data_handler = load(model_path)
        predictor = JointPredictor(model, data_handler)
        predictor_input = pd.DataFrame({"text": [test_text], "dict_feat": [test_dict]})

        predictions = predictor.predict(predictor_input)
        self.assertEqual(predictions[0][0]["name"], "word_scores:PAD_LABEL")
        np.testing.assert_array_almost_equal(
            predictions[0][0]["value"],
            [
                -22.83,
                -23.99,
                -31.9,
                -24.43,
                -14.11,
                -14.49,
                -6.26,
                -7.51,
                -22.68,
                -10.69,
                -10.67,
                -23.79,
            ],
            decimal=2,
        )
