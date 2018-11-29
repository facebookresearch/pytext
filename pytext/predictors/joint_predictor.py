#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.common.constants import DatasetFieldName
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.predictors.classifier_predictor import ClassifierPredictor
from pytext.predictors.predictor import Predictor
from pytext.predictors.tagger_predictor import TaggerPredictor


class JointPredictor(Predictor):
    def fill_predictions(
        self, model_output: List[Any], context: Dict[str, Any]
    ) -> List[List[Dict]]:
        doc_label_names = self.data_handler.metadata.labels[
            DatasetFieldName.DOC_LABEL_FIELD
        ].vocab.itos
        word_label_names = self.data_handler.metadata.labels[
            DatasetFieldName.WORD_LABEL_FIELD
        ].vocab.itos

        word_predictions = TaggerPredictor.fill_tagger_predictions(
            word_label_names,
            model_output[1],
            context[DatasetFieldName.INDEX_FIELD],
            context[DatasetFieldName.TOKEN_RANGE],
            context[SEQ_LENS],
            "word_scores",
        )
        doc_predictions = ClassifierPredictor.fill_classifier_predictions(
            doc_label_names,
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
