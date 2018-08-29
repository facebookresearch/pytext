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
                -31.65,
                -33.92,
                -38.48,
                -26.44,
                -11.58,
                -19.31,
                -6.6,
                -6.82,
                -23.79,
                -9.98,
                -13.2,
                -22.27,
            ],
            decimal=2,
        )
