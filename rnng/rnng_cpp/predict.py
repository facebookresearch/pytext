#!/usr/bin/env python3

import argparse
import datetime
import torch
from pytext.shared_tokenizer import SharedTokenizer
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.rnng.rnng_cpp.caffe2_utils import read_pytorch_model, export_model
from rnng_bindings import Parser


class RNNGPredictor_CPP:
    def __init__(self, model_path: str) -> None:

        self.tokenizer = SharedTokenizer()
        self.featurizer = SharedFeaturizer()
        pytorch_config = read_pytorch_model(model_path)

        self.model = Parser(
            pytorch_config["model_config"],
            pytorch_config["actions_vec"],
            pytorch_config["terminals_vec"],
            pytorch_config["dictfeats_vec"],
        ).make()

        self.model.load_state_dict(pytorch_config["model_state"], False)
        print("Loaded model")

        self.model.eval()

        # Workaround for eval_rnng. Will be cleanly removed in later refactor
        snapshot = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actions_bidict = snapshot["oracle_dicts"].actions_bidict
        self.terminal_bidict = snapshot["oracle_dicts"].terminal_bidict
        self.dictfeat_bidict = snapshot["oracle_dicts"].dictfeat_bidict

    def predict(self, text: str, sparsefeat: str) -> str:

        model_feats = self.featurizer.featurize(text, sparsefeat)

        return self.model.predict(
            self.tokenizer.tokenize(text),
            model_feats.dictFeats,
            model_feats.dictFeatWeights,
            model_feats.dictFeatLengths,
        )


def predict_on_file(predictor, utt_file, output_file):
    ttime = 0
    num_queries = 1
    with open(utt_file) as f, open(output_file, "w") as wf:
        for line in f:
            start_time = datetime.datetime.now()
            pred = predictor.predict(line, "{}")  # TODO: Add local sparsefeat reading
            time_taken = datetime.datetime.now() - start_time
            ttime += time_taken.microseconds / 1000
            wf.write(pred + "\n")
            num_queries += 1
    print("Time per query:{}".format(ttime / num_queries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using RNNG model")
    parser.add_argument(
        "-model_path", type=str, help="model path", default="/tmp/model.pt"
    )
    parser.add_argument("-text", type=str, help="text to parse", default="")
    parser.add_argument(
        "-sparsefeat", type=str, help="raw dictionary features", default="{}"
    )
    parser.add_argument("-caffe2_model_path", type=str, default="")
    parser.add_argument("-input_file", type=str, help="input file")
    parser.add_argument("-output_file", type=str, help="output file")

    args = parser.parse_args()

    predictor = RNNGPredictor_CPP(args.model_path)
    if torch.cuda.is_available():
        print("Running on GPU")
        predictor.model.cuda()

    if args.text:
        print(predictor.predict(args.text, args.sparsefeat))

    if args.input_file and args.output_file:
        predict_on_file(predictor, args.input_file, args.output_file)

    if args.caffe2_model_path:
        export_model(args.model_path, args.caffe2_model_path)
        print("Exported Caffe2 model to", args.caffe2_model_path)
