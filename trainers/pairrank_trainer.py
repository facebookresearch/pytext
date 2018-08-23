#!/usr/bin/env python3

import sys
import torch
import numpy as np
from pytext.loss.pairrank_loss import PairRankLoss
from pytext.utils import test_utils
from pytext.utils import cuda_utils
from collections import namedtuple
from .trainer import Trainer


PairNNBatch = namedtuple("PairNNBatch", ["query_text", "results"])


class PairRankTrainer(Trainer):
    def __init__(self, loss: PairRankLoss, config, **kwargs) -> None:
        assert isinstance(loss, PairRankLoss)
        super().__init__(config)

    def process_batch(self, batch, model):
        targets = cuda_utils.Variable(torch.FloatTensor([1] * batch.batch_size))
        return (
            (PairNNBatch(batch.query_text, [batch.pos_text, batch.neg_text]),),
            None,
            targets,
        )

    def report(self, stage, loss, preds, seq_lens, target, target_names):
        # TODO:add more detailed reports later for ranking loss
        sys.stdout.write("{} - loss: {:.6f}\n".format(stage, loss))
        return loss

    def evaluate(self, dev_iter, model, class_names):
        model.eval()
        losses = []
        for batch in dev_iter:
            m_input, targets = self.process_batch(batch, model)
            m_out = model(*m_input)
            losses.append(self.loss(m_out, targets).data[0])
        model.train()

        total_loss = np.mean(losses)
        model.train()
        self.report("Evaluation", total_loss, None, None, None)
        return total_loss

    def test(self, model, test_ds, test_iter, class_names):
        model.eval()
        metrics_dict = {
            "num_samples": 0,
            "num_correct_comparisons": 0,
            "sum_dist_between_pos_neg": 0.0,
        }
        for batch in test_iter:
            m_input, targets = self.process_batch(batch, model)
            m_out = model(*m_input)
            self.update_test_results(m_out, metrics_dict)

        metric_to_report = {
            "num_samples": metrics_dict["num_samples"],
            "num_correct_comparisons": metrics_dict["num_correct_comparisons"],
            "precision": metrics_dict["num_correct_comparisons"]
            * 1.0
            / metrics_dict["num_samples"],
            "average_dist_between_pos_neg": metrics_dict["sum_dist_between_pos_neg"]
            * 1.0
            / metrics_dict["num_samples"],
        }
        return [], None, test_utils.ResultRow("metrics", metric_to_report)

    def update_test_results(self, m_out, metrics_dict):
        pos_similarity, neg_similarity = (cuda_utils.var_to_numpy(s[0]) for s in m_out)

        metrics_dict["num_samples"] += np.size(pos_similarity)
        metrics_dict["num_correct_comparisons"] = np.sum(
            pos_similarity > neg_similarity
        )
        metrics_dict["sum_dist_between_pos_neg"] += np.sum(
            pos_similarity - neg_similarity
        )
