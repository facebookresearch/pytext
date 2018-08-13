#!/usr/bin/env python3

from pytext.predictors.predictor import Predictor
from pytext.predictors.classifier_predictor import ClassifierPredictor
from pytext.predictors.tagger_predictor import TaggerPredictor
from typing import List, Any, Dict
from pytext.common.constants import DatasetFieldName
from pytext.data.joint_data_handler import SEQ_LENS


class JointPredictor(Predictor):
    def fill_predictions(
        self, model_output: List[Any], context: Dict[str, Any]
    ) -> List[List[Dict]]:
        [doc_class_names, word_class_names] = self.data_handler.metadata["class_names"]
        word_predictions = TaggerPredictor.fill_tagger_predictions(
            word_class_names,
            model_output[1],
            context[DatasetFieldName.INDEX_FIELD],
            context[DatasetFieldName.TOKEN_RANGE_PAIR],
            context[SEQ_LENS],
            "word_scores",
        )
        doc_predictions = ClassifierPredictor.fill_classifier_predictions(
            doc_class_names,
            model_output[0],
            context[DatasetFieldName.INDEX_FIELD],
            "doc_scores",
        )
        # Merge word and doc predictions into one joint list
        joint_predictions = [
            word_predictions[i] + doc_predictions[i]
            for i in range(len(word_predictions))
        ]
        return joint_predictions
