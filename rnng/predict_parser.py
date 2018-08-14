#!/usr/bin/env python3
import numpy as np
import os
import torch
import argparse
import datetime
import pytext.rnng.read_data as read_data
import pytext.utils.cuda_utils as cuda_utils
from typing import List, Tuple, Union
from pytext.rnng.Parser import RNNGParser
from pytext.rnng.utils import BiDict, NUM, is_number
from assistant.lib.feat.ttypes import ModelFeatures
from pytext.config import PyTextConfig, config_from_json
from pytext.shared_tokenizer import SharedTokenizer
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.rnng.annotation import list_from_actions, Tree, tree_from_actions

# need this because of a bug in PyTorch (?) which makes things go awry
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

cuda_utils.CUDA_ENABLED = torch.cuda.is_available()

tokenizer = SharedTokenizer()

EMPTY_BIDICT = BiDict()
EMPTY_LIST_FEAT: List = []
EMPTY_LIST_WT: List = []
EMPTY_LIST_LEN: List = []


class RNNGPredictor:
    def __init__(
        self,
        rnng_model: RNNGParser,
        terminal_bidict: BiDict,
        actions_bidict: BiDict,
        dictfeat_bidict: BiDict = EMPTY_BIDICT,
    ) -> None:
        self.model = rnng_model
        self.model.eval()
        self.terminal_bidict = terminal_bidict
        self.actions_bidict = actions_bidict
        self.dictfeat_bidict = dictfeat_bidict

    def preprocess(self, tokens_str: List[str]) -> List[int]:
        tokens_str = [NUM if is_number(t) else t.lower() for t in tokens_str]
        tokens = [read_data.unkify(t, self.terminal_bidict) for t in tokens_str]
        return read_data.indices(tokens, self.terminal_bidict)

    def predict_idx(
        self,
        tokens_str: List[str],
        dict_feats: List[str] = EMPTY_LIST_FEAT,
        dict_feat_weights: List[float] = EMPTY_LIST_WT,
        dict_feat_lengths: List[int] = EMPTY_LIST_LEN,
        beam_size: int = 1,
    ) -> Tuple[List[int], List[float]]:
        """
        Returns:
            A (int) list of actions taken (indices into actions_bidict)
            A (float) list of confidence score of those actions,
                computed from the softmax output
        """
        tokens_indices = self.preprocess(tokens_str)
        dict_feat_indices = read_data.indices(dict_feats, self.dictfeat_bidict)
        predicted_idx, predicted_all_scores = self.model.forward(
            [
                cuda_utils.Variable(torch.LongTensor(tokens_indices[::-1])),
                cuda_utils.Variable(torch.LongTensor(dict_feat_indices[::-1])),
                cuda_utils.Variable(torch.FloatTensor(dict_feat_weights[::-1])),
                cuda_utils.Variable(torch.LongTensor(dict_feat_lengths[::-1])),
            ],
            beam_size,
        )
        predicted_scores = [
            np.exp(np.max(action_scores)).item() / np.sum(np.exp(action_scores)).item()
            for action_scores in predicted_all_scores.detach().numpy()
        ]
        return predicted_idx.numpy(), predicted_scores

    def predict_actions_and_tree(
        self,
        tokens_str: List[str],
        dict_feats: List[str],
        dict_feat_weights: List[float],
        dict_feat_lengths: List[int],
        beam_size: int = 1,
    ) -> Tuple[List[int], Tree, List[float]]:
        predicted_idx, predicted_scores = self.predict_idx(
            tokens_str, dict_feats, dict_feat_weights, dict_feat_lengths, beam_size
        )
        return (
            list_from_actions(tokens_str, self.actions_bidict, predicted_idx),
            tree_from_actions(tokens_str, self.actions_bidict, predicted_idx),
            predicted_scores,
        )

    def predict_actions(
        self,
        tokens_str: List[str],
        dict_feats: List[str],
        dict_feat_weights: List[float],
        dict_feat_lengths: List[int],
        beam_size: int = 1,
    ) -> List[int]:
        return self.predict_actions_and_tree(
            tokens_str, dict_feats, dict_feat_weights, dict_feat_lengths, beam_size
        )[0]

    def predict(
        self,
        tokens_str: List[str],
        dict_feats: List[str],
        dict_feat_weights: List[float],
        dict_feat_lengths: List[int],
        beam_size: int = 1,
    ) -> Tree:
        return self.predict_actions_and_tree(
            tokens_str, dict_feats, dict_feat_weights, dict_feat_lengths, beam_size
        )[1]


def load_model(model_snapshot_path: str) -> RNNGPredictor:
    snapshot = torch.load(
        model_snapshot_path, map_location=lambda storage, loc: storage
    )
    model_state = snapshot["model_state"]
    pytext_config = config_from_json(PyTextConfig, snapshot["pytext_config"])
    oracle_dicts = snapshot["oracle_dicts"]

    rnng_model = RNNGParser(
        pytext_config,
        oracle_dicts.terminal_bidict,
        oracle_dicts.actions_bidict,
        oracle_dicts.dictfeat_bidict,
    )
    rnng_model.load_state_dict(model_state)
    print("Loaded model")

    if cuda_utils.CUDA_ENABLED:
        print("Running on GPU")
        rnng_model.cuda()

    return RNNGPredictor(
        rnng_model,
        oracle_dicts.terminal_bidict,
        oracle_dicts.actions_bidict,
        oracle_dicts.dictfeat_bidict,
    )


def get_pred_input(raw_input: Union[str, List[str], Tuple[str]], add_dict_feat=False):
    if isinstance(raw_input, str):
        parts = raw_input.split("\t")
    elif isinstance(raw_input, (list, tuple)):
        parts = list(raw_input)
    else:
        raise ValueError(
            "Unexpected raw_input type: {}. Should be str, list or tuple.".format(
                type(raw_input)
            )
        )

    utterance, sparse_feat = None, None
    if len(parts) == 1:
        [utterance] = parts
    elif len(parts) == 2:  # Assuming line = "utterance\tdict_feat"
        [utterance, sparse_feat] = parts
    else:
        raise ValueError("Unexpected input for predictor: {}".format(raw_input))

    toks = tokenizer.tokenize(utterance)
    dict_feats: List[str] = []
    dict_feat_weights: List[float] = []
    dict_feat_lengths: List[int] = []
    if add_dict_feat:
        model_feats: ModelFeatures = SharedFeaturizer().featurize(
            utterance, sparse_feat
        )
        dict_feats = model_feats.dictFeats
        dict_feat_weights = model_feats.dictFeatWeights
        dict_feat_lengths = model_feats.dictFeatLengths

    return toks, dict_feats, dict_feat_weights, dict_feat_lengths


def predict_actions_and_tree(
    predictor: RNNGPredictor,
    raw_input: Union[str, List[str], Tuple[str]],
    add_dict_feat: bool = False,
):
    (toks, dict_feats, dict_feat_weights, dict_feat_lengths) = get_pred_input(
        raw_input, add_dict_feat
    )
    return predictor.predict_actions_and_tree(
        toks, dict_feats, dict_feat_weights, dict_feat_lengths
    )


def predict(
    predictor: RNNGPredictor,
    raw_input: Union[str, List[str], Tuple[str]],
    add_dict_feat: bool = False,
):
    (toks, dict_feats, dict_feat_weights, dict_feat_lengths) = get_pred_input(
        raw_input, add_dict_feat
    )
    return predictor.predict(toks, dict_feats, dict_feat_weights, dict_feat_lengths)


def predict_on_file(
    predictor,
    utt_file,
    output_file,
    beam_size=1,
    tagged_input=False,
    add_dict_feat=False,
):
    ttime = 0
    num_queries = 0
    with open(utt_file) as f, open(output_file, "w") as wf:
        if tagged_input:
            f = read_data.read_annotated_file(
                utt_file, predictor.actions_bidict, add_dict_feat=add_dict_feat
            )
        for line in f:
            start_time = datetime.datetime.now()
            if tagged_input:
                sent = line.sentence
                toks = sent.raw
                dict_feats = sent.dict_feats
                dict_feat_weights = sent.dict_feat_weights
                dict_feat_lengths = sent.dict_feat_lengths
            else:
                (
                    toks,
                    dict_feats,
                    dict_feat_weights,
                    dict_feat_lengths,
                ) = get_pred_input(line, add_dict_feat=add_dict_feat)
            tree = predictor.predict(
                toks, dict_feats, dict_feat_weights, dict_feat_lengths, beam_size
            )
            time_taken = datetime.datetime.now() - start_time
            ttime += time_taken.microseconds / 1000
            wf.write(tree.flat_str() + "\n")
            num_queries += 1
    print("Time per query:{}".format(ttime / num_queries))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description="Predict using RNNG model")
    parser.add_argument("-model_path", type=str, help="model path")
    parser.add_argument("-input_file", type=str, help="input file")
    parser.add_argument("-output_file", type=str, help="output file")
    parser.add_argument("-seed", type=int, help="RNG seed", default=37)
    parser.add_argument("-beam_size", type=int, help="decode beam size", default=1)
    parser.add_argument("-tagged_input", type=bool, help="input format", default=False)
    parser.add_argument(
        "-add_dict_feat", type=bool, help="use dict feats", default=False
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    predictor = load_model(args.model_path)
    predict_on_file(
        predictor,
        args.input_file,
        args.output_file,
        args.beam_size,
        args.tagged_input,
        args.add_dict_feat,
    )
