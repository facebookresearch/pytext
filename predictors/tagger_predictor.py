#!/usr/bin/env python3

from pytext.predictors.predictor import Predictor
from typing import List, Any, Tuple, Dict
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.data.joint_data_handler import SEQ_LENS


class TaggerPredictor(Predictor):
    def fill_predictions(
        self, model_output: List[Any], context: Dict[str, Any]
    ) -> List[Any]:
        [label_meta] = self.data_handler.metadata.labels.values()
        return TaggerPredictor.fill_tagger_predictions(
            label_meta.vocab.itos,
            model_output[0],
            context[DatasetFieldName.INDEX_FIELD],
            context[DatasetFieldName.TOKEN_RANGE_PAIR],
            context[SEQ_LENS],
        )

    @staticmethod
    def fill_tagger_predictions(
        tagger_classes: List[str],
        tagger_output: Any,
        orig_input_indices: List[int],
        tokenized_tokens: List[List[Tuple[str, Tuple[int, int]]]],
        ex_lengths: List[int],
        name_prefix: str = "scores",
    ) -> List[Any]:
        predictions: List[Any] = [None] * len(orig_input_indices)
        prediction_names = [
            "{}:{}".format(name_prefix, class_name) for class_name in tagger_classes
        ]
        padded_output = F.log_softmax(tagger_output, 2)
        for i, ex_model_out in enumerate(padded_output.data):
            prediction_pairs = [
                {"name": pred_pair[0], "value": list(map(float, pred_pair[1]))}
                for pred_pair in zip(
                    prediction_names, ex_model_out.t_()[:, : ex_lengths[i]]
                )
            ]
            tokens = [tok for (tok, (_, _)) in tokenized_tokens[i]]
            token_ranges = list(
                sum([t_range for (_, t_range) in tokenized_tokens[i]], ())
            )
            prediction_pairs.append({"name": "tokens", "value": tokens})
            prediction_pairs.append({"name": "token_ranges", "value": token_ranges})

            predictions[orig_input_indices[i]] = prediction_pairs

        return predictions
