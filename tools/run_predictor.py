#!/usr/bin/env python3

import os
import argparse
from pytext.predictors import load_predictor
import pandas as pd


def run_predictor(model_path: str, text: str, dict_feat: str) -> None:
    if model_path is None or not os.path.isfile(model_path):
        raise ValueError("Invalid snapshot path for testing")

    predictor = load_predictor(model_path)
    predictor_input = pd.DataFrame({"text": [text], "dict_feat": [dict_feat]})
    print(predictor.predict(predictor_input))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-file", default="/tmp/model.pt", type=str)
    parser.add_argument("--text", default="", type=str)
    parser.add_argument("--dict-feat", default="", type=str)
    args = parser.parse_args()
    run_predictor(args.predictor_file, args.text, args.dict_feat)
