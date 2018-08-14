#!/usr/bin/env python3

import json
import sys

import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.common.registry import TRAINER, component
from pytext.config.pytext_config import ConfigBase
from pytext.utils import test_utils
from sklearn.metrics import classification_report, f1_score

from .trainer import Trainer, TrainerConfig


class ClassifierTrainerConfig(ConfigBase, TrainerConfig):
    pass


@component(TRAINER, config_cls=ClassifierTrainerConfig)
class ClassifierTrainer(Trainer):

    def report(self, stage, loss, preds, seq_lens, target, target_names):
        [target], [preds] = target, preds
        [target_names] = target_names

        sys.stdout.write("{} - loss: {:.6f}\n".format(stage, loss))
        sys.stdout.write(
            classification_report(target.data, preds, target_names=target_names)
        )
        return f1_score(target.data, preds, average="weighted")

    def test(self, model, test_iter, metadata):
        model.eval()
        [class_names] = metadata["class_names"]

        # Write header lines
        preds_table = []
        preds_table.append("#{0}".format(json.dumps(class_names)))
        preds_table.append(("#predictions", "label", "doc_index", "scores", "text"))
        all_targets, all_preds = None, None

        for m_input, [targets], context in test_iter:

            [m_out] = model(*m_input)
            preds = torch.max(m_out, 1)[1].data
            self.update_test_results(
                preds_table,
                preds,
                targets.data,
                m_out,
                class_names,
                context[DatasetFieldName.TOKEN_RANGE_PAIR],
                context[DatasetFieldName.INDEX_FIELD],
                context[DatasetFieldName.UTTERANCE_FIELD],
            )
            if all_targets is None:
                all_preds = preds
                all_targets = targets
            else:
                all_targets = torch.cat((all_targets, targets), 0)
                all_preds = torch.cat((all_preds, preds), 0)

        result_table, weighted_metrics = test_utils.get_all_metrics(
            all_preds, all_targets.data, class_names
        )
        # TODO: define frame metrics
        return preds_table, result_table, weighted_metrics, None

    def update_test_results(
        self,
        preds_table,
        preds,
        labels_idx,
        m_out,
        class_names,
        token_range_pair,
        orig_indices,
        utterance,
    ):
        for i in range(len(token_range_pair)):
            preds_table.append(
                (
                    preds[i].item(),
                    labels_idx[i].item(),
                    orig_indices[i],
                    ",".join(
                        ["{0:.2f}".format(_s) for _s in F.softmax(m_out[i], 0).data]
                    ),
                    utterance[i],
                )
            )
