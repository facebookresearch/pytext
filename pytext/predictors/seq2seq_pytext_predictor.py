#!/usr/bin/env python3
from typing import Any, Dict, List

import torch
from fairseq import models, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from pytext.common.constants import DatasetFieldName
from torch.autograd import Variable


class SEQ2SEQPyTextPredictor:
    def __init__(self, predictor_state) -> None:
        self.use_cuda = torch.cuda.is_available()
        model_args = predictor_state["args"]

        self.src_dict = predictor_state["src_dict"]
        self.dst_dict = predictor_state["dst_dict"]
        model = models.build_model(model_args, self.src_dict, self.dst_dict)
        model.load_state_dict(predictor_state["model"])

        # TODO: Add beam size and other parameters to config
        model.make_generation_fast_(beamable_mm_beam_size=5)

        ensemble = [model]

        # Initialize generator
        self.translator = SequenceGenerator(
            ensemble, beam_size=5, stop_early=True, normalize_scores=True, unk_penalty=2
        )
        if self.use_cuda:
            self.translator.cuda()

    def predict(self, data: List[Dict[str, Any]]) -> List[List[Dict]]:
        predicts = []
        # Most of the code below is copy pasta from fairseq-py/interactive.py
        for utterance in data[DatasetFieldName.TEXT_FIELD]:
            src_str = utterance.strip()
            src_tokens = tokenizer.Tokenizer.tokenize(
                src_str, self.src_dict, add_if_not_exist=False
            ).long()
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
            src_lengths = src_tokens.new([src_tokens.numel()])
            translations = self.translator.generate(
                Variable(src_tokens.view(1, -1)), Variable(src_lengths.view(-1))
            )
            hypos = translations[0]
            prediction_pairs = []

            # Process top predictions
            for hypo in hypos[: min(len(hypos), 1)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"].int().cpu(),
                    dst_dict=self.dst_dict,
                    align_dict=None,
                    remove_bpe=None,
                )
                prediction_pairs.append({"name": "hypothesis", "value": hypo_str})

            predicts.append(prediction_pairs)

        return predicts
