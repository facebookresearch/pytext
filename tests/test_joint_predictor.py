#!/usr/bin/env python3

import unittest

import numpy as np
import pandas as pd
from pytext.jobspec import register_builtin_jobspecs
from pytext.predictors.joint_predictor import JointPredictor
from pytext.serialize import load


register_builtin_jobspecs()


class JointPredictorTest(unittest.TestCase):
    def test_predict(self):
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
                -26.85,
                -29.04,
                -31.91,
                -27.31,
                -14.74,
                -15.54,
                -7.82,
                -8.76,
                -25.57,
                -9.88,
                -13.41,
                -23.02,
            ],
            decimal=2,
        )
