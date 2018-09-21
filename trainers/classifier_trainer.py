#!/usr/bin/env python3

import json
import sys
from typing import List

import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.metrics import (
    LabelPredictionPair,
    LabelScoresPair,
    compute_classification_metrics,
)
from sklearn.metrics import classification_report, f1_score

from .trainer import Trainer


class ClassifierTrainer(Trainer):
    def report(self, stage, loss, preds, seq_lens, targets, target_names):
        [target_names] = target_names
        pred = torch.cat([p for p in preds], 0)
        target = torch.cat([t for t in targets], 0)
        sys.stdout.write("{} - loss: {:.6f}\n".format(stage, loss))
        sys.stdout.write(
            classification_report(target.cpu(), pred.cpu(), target_names=target_names)
        )
        return f1_score(target.cpu(), pred.cpu(), average="weighted")

    def test(self, model, test_iter, metadata):
        model.eval()
        [label_meta] = metadata.labels.values()
        label_names = label_meta.vocab.itos
        # Write header lines
        preds_table = []
        label_pairs: List[LabelPredictionPair] = []
        label_scores: List[LabelScoresPair] = []
        preds_table.append("#{0}".format(json.dumps(label_names)))
        preds_table.append(("#predictions", "label", "doc_index", "scores", "text"))
        all_targets, all_preds = None, None

        for m_input, [targets], context in test_iter:

            m_out = model(*m_input)
            preds = torch.max(m_out, 1)[1].data
            self.update_test_results(
                preds_table,
                preds,
                targets.data,
                m_out,
                label_names,
                context[DatasetFieldName.TOKEN_RANGE_PAIR],
                context[DatasetFieldName.INDEX_FIELD],
                context[DatasetFieldName.UTTERANCE_FIELD],
                label_pairs,
                label_scores,
            )
            if all_targets is None:
                all_preds = preds
                all_targets = targets
            else:
                all_targets = torch.cat((all_targets, targets), 0)
                all_preds = torch.cat((all_preds, preds), 0)

        metrics = compute_classification_metrics(label_pairs, label_names, label_scores)
        return preds_table, metrics

    def update_test_results(
        self,
        preds_table,
        preds,
        labels_idx,
        m_out,
        label_names,
        token_range_pair,
        orig_indices,
        utterances,
        label_pairs,
        label_scores,
    ):
        for i in range(len(token_range_pair)):
            # TODO scores can be get from model now, will refactor in next diff
            scores = F.softmax(m_out[i], 0).data
            preds_table.append(
                (
                    preds[i].item(),
                    labels_idx[i].item(),
                    orig_indices[i],
                    ",".join(["{0:.2f}".format(_s) for _s in scores]),
                    utterances[i],
                )
            )

            predicted_label = label_names[preds[i].item()]
            expected_label = label_names[labels_idx[i].item()]
            label_pairs.append(LabelPredictionPair(predicted_label, expected_label))
            label_scores.append(LabelPredictionPair(scores, expected_label))
