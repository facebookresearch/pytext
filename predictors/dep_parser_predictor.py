#!/usr/bin/env python3

from .predictor import Predictor
from typing import List, Dict
import pandas as pd


class DepParserPredictor(Predictor):
    def predict(self, df: pd.DataFrame) -> List[List[Dict]]:
        examples_texts = df["text"].tolist()
        model_out = self.model(examples_texts)
        predictions = []
        for ex_result in model_out:
            prediction_pairs = []
            prediction_pairs.append(
                {
                    "name": "tokens",
                    "value": [tok for (tok, (_, _)) in ex_result.tokens_with_ranges],
                }
            )
            prediction_pairs.append(
                {
                    "name": "token_ranges",
                    "value": list(
                        sum(
                            [t_range for (_, t_range) in ex_result.tokens_with_ranges],
                            (),
                        )
                    ),
                }
            )
            prediction_pairs.append(
                {"name": "pos_tags", "value": [tag.label for tag in ex_result.pos_tags]}
            )
            prediction_pairs.append(
                {
                    "name": "head_indices",
                    "value": [
                        ex_result.dependency_parse[i].head_idx
                        for i in range(len(ex_result.tokens_with_ranges))
                    ],
                }
            )
            prediction_pairs.append(
                {
                    "name": "dependencies",
                    "value": [
                        ex_result.dependency_parse[i].dep
                        for i in range(len(ex_result.tokens_with_ranges))
                    ],
                }
            )
            predictions.append(prediction_pairs)

        return predictions
