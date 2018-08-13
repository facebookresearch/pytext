#!/usr/bin/env python3

from .docnn import DocNN
from pytext.config.ttypes import PairNNSubModel
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn

from collections import namedtuple

FakeBatch = namedtuple("FakeBatch", ["text"])


class PairNN(nn.Module):
    def __init__(self, config):
        super(PairNN, self).__init__()
        pairnn_config = config.model.value
        self.config = pairnn_config

        def _create_sub_model(config):
            if config.model.getType() == PairNNSubModel.DOCNN:
                return DocNN(config)
            else:
                raise ValueError(
                    "Model type {} not suppoert in PairNN".format(
                        config.model.getType()
                    )
                )

        def _build_sub_nn(config, model_config, embed_size):
            sub_config = deepcopy(config)
            sub_config.model = model_config
            sub_config.runtime_params.doc_class_num = embed_size
            return _create_sub_model(sub_config)

        self.query_nn = _build_sub_nn(
            config, pairnn_config.query_model, pairnn_config.embed_size
        )
        self.result_nn = _build_sub_nn(
            config, pairnn_config.result_model, pairnn_config.embed_size
        )

    def _create_fake_batch(self, b):
        """
        b is a namedtuple with attrs {'query_text','results'}. b.results is a
        list of docs to be evaluated together with b.query_text. During training,
        b.results = [pos_text, neg_text]
        """
        return FakeBatch(b.query_text), [FakeBatch(result) for result in b.results]

    def forward(self, b):
        b_query, b_results = self._create_fake_batch(b)
        [query_logit] = self.query_nn(*self.query_nn.unpack_batch(b_query))
        result_logits = [
            self.result_nn(*self.result_nn.unpack_batch(b_result))[0]
            for b_result in b_results
        ]
        return (
            [F.cosine_similarity(query_logit, result_logit)]
            for result_logit in result_logits
        )
