#!/usr/bin/env python3

import sys
import torch
from pytext.utils.cuda_utils import Variable
from pytext.common.constants import Padding, DatasetFieldName
from pytext.data.joint_data_handler import SEQ_LENS
from sklearn.metrics import classification_report, f1_score
from pytext.utils import test_utils
from pytext.utils.data_utils import Slot
from pytext.common.registry import TRAINER, component
from pytext.config.pytext_config import ConfigBase
import json
from .trainer import Trainer, TrainerConfig


class TaggerTrainerConfig(ConfigBase, TrainerConfig):
    pass


@component(TRAINER, config_cls=TaggerTrainerConfig)
class TaggerTrainer(Trainer):
    def report(self, stage, loss, preds, seq_lens, target, target_names):
        [target], [preds] = target, preds
        [target_names] = target_names
        preds, target = TaggerTrainer.remove_padding(preds, target)
        sys.stdout.write("{} - loss: {:.6f}\n".format(stage, loss))
        sys.stdout.write(
            classification_report(
                target.cpu(),
                preds.cpu(),
                target_names=target_names[Padding.WORD_LABEL_PAD_IDX + 1 :],
            )
        )
        return f1_score(target.data.cpu(), preds.cpu(), average="weighted")

    def test(self, model, test_iter, metadata):
        model.eval()

        preds_table = []
        [word_class_names] = metadata["class_names"]
        word_class_names, mapping = TaggerTrainer.filter_word_labels(word_class_names)

        preds_table.append("#{0}".format(json.dumps(word_class_names)))
        preds_table.append(
            (
                "#predictions",
                "label",
                "doc_index",
                "[pred:lab]",
                "tok",
                "text",
                "chunk_match",
            )
        )
        all_targets = None
        all_preds = None
        for m_input, [targets], context in test_iter:
            [m_out] = model(*m_input)
            if hasattr(model, "crf") and model.crf:
                m_out = model.crf.decode_crf(m_out, targets)

            m_out = self._flatten_2d(m_out)
            targets = targets.view(-1)
            preds = torch.max(m_out, 1)[1].data
            preds, targets = TaggerTrainer.remove_padding(preds, targets)
            preds = TaggerTrainer.map_to_filtered_ids(preds, mapping)
            targets = TaggerTrainer.map_to_filtered_ids(targets, mapping)

            self.update_test_results(
                preds_table,
                preds,
                targets,
                word_class_names,
                context[SEQ_LENS],
                context[DatasetFieldName.RAW_WORD_LABEL],
                context[DatasetFieldName.TOKEN_RANGE_PAIR],
                context[DatasetFieldName.INDEX_FIELD],
            )
            if all_preds is None:
                all_preds = preds
                all_targets = targets
            else:
                all_preds = torch.cat((all_preds, preds), 0)
                all_targets = torch.cat((all_targets, targets), 0)

        result_table, weighted_metrics = test_utils.get_all_metrics(
            all_preds.cpu(), all_targets.cpu(), word_class_names
        )
        # TODO: define frame metrics
        return preds_table, result_table, weighted_metrics, None

    def update_test_results(
        self,
        test_results,
        preds,
        targets,
        class_names,
        seq_lens,
        raw_labels,
        tokenized_examples,
        orig_indices,
    ):
        offset = 0
        for i in range(seq_lens.size()[0]):
            preds_idx = preds[offset : offset + seq_lens[i]]
            target_idx = targets[offset : offset + seq_lens[i]]
            offset += seq_lens[i]
            preds_names = [class_names[p] for p in preds_idx]
            label_names = raw_labels[i]
            preds_names = test_utils.summarize(
                seq_lens[i], tokenized_examples[i], preds_names
            )
            pred_lab = ":".join(
                [str(list(map(int, preds_idx))), str(list(map(int, target_idx)))]
            )
            tokens = [t for t, _ in tokenized_examples[i]]
            test_results.append(
                (
                    preds_names,
                    label_names,
                    orig_indices[i],
                    pred_lab,
                    tokens,
                    " ".join(tokens),
                    test_utils.count_chunk_match(preds_names, label_names),
                )
            )

    @staticmethod
    def remove_padding(preds, targets):
        filter_idx = Variable(
            torch.LongTensor(
                [
                    i
                    for i, target in enumerate(targets)
                    if target.item() != Padding.WORD_LABEL_PAD_IDX
                ]
            )
        )
        return preds[filter_idx], targets[filter_idx]

    @staticmethod
    def filter_word_labels(word_class_names):
        # Filters Padding token and BIO prefix
        filtered_names = []
        mapping = {}
        for i, c_name in enumerate(word_class_names):
            if c_name == Padding.WORD_LABEL_PAD:
                continue

            to_strip = 0
            if c_name.startswith(Slot.B_LABEL_PREFIX) or c_name.startswith(
                Slot.I_LABEL_PREFIX
            ):
                to_strip = len(Slot.B_LABEL_PREFIX)
            c_name = c_name[to_strip:]

            if c_name not in filtered_names:
                filtered_names.append(c_name)

            mapping[i] = filtered_names.index(c_name)

        return filtered_names, mapping

    @staticmethod
    def map_to_filtered_ids(orig_ids, mapping):
        return Variable(torch.LongTensor([mapping[i.item()] for i in orig_ids]))
