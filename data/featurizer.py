#!/usr/bin/env python3
from enum import Enum
from joblib import Parallel, delayed
from typing import Any, Dict, Sequence

class InputKeys(Enum):
    RAW_TEXT = "raw_text"
    TOKEN_FEATURES = "raw_dict"
    LOCALE = "locale"

class OutputKeys(Enum):
    FEATURES = "features_obj"
    TOKENIZED_TEXT = "tokenized_text"


class Featurizer(object):

    def featurize(self, input_record: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Featurizer.featurize() method must be implemented.")

    def featurize_batch(
        self,
        input_record_list: Sequence[Dict[str, Any]],
    ) -> Sequence[Dict[str, Any]]:
        """Featurize a batch of instances/examples in parallel.
        This is a default implementation using joblib.Parallel,
        feel free to re-implement it as needed.
        """
        features_list = Parallel(n_jobs=-1)(
            delayed(self.featurize)(record)
            for record in input_record_list
        )
        return features_list
