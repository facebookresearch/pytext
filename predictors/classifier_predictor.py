#!/usr/bin/env python3

import torch.nn.functional as F
from pytext.predictors.predictor import Predictor
from pytext.utils.cuda_utils import Variable
from pytext.common.constants import DatasetFieldName
from typing import List, Any, Dict


class ClassifierPredictor(Predictor):
    def fill_predictions(
        self, model_output: List[Any], context: Dict[str, Any]
    ) -> List[Any]:
        return ClassifierPredictor.fill_classifier_predictions(
            self.data_handler.metadata["class_names"][0],
            model_output[0],
            context[DatasetFieldName.INDEX_FIELD],
        )

    @staticmethod
    def fill_classifier_predictions(
        classifier_classes: List[str],
        classifier_output: Any,
        orig_input_indices: List[int],
        name_prefix: str = "scores",
    ) -> List[Any]:
        predictions: List[Any] = [None] * len(classifier_output)
        prediction_names = [
            "{}:{}".format(name_prefix, class_name) for class_name in classifier_classes
        ]
        predictions
        for i, ex_model_out in enumerate(classifier_output.data):
            prediction_pairs = [
                {"name": pred_pair[0], "value": [pred_pair[1].item()]}
                for pred_pair in zip(
                    prediction_names, F.log_softmax(Variable(ex_model_out), 0).data
                )
            ]
            predictions[i] = prediction_pairs

        return predictions
