#!/usr/bin/env python3

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName, Padding
from pytext.config.pytext_config import ConfigBase
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.metrics import Node, Span, compute_all_metrics
from pytext.utils import data_utils, test_utils
from sklearn.metrics import classification_report

from .tagger_trainer import TaggerTrainer
from .trainer import Trainer


class JointTrainer(Trainer):
    # TODO: (wenfangxu) T32687556 define unified interfaces for report() and test()
    def report(self, stage, loss, preds, seq_lens, targets, target_names):
        d_target, w_target = targets
        d_preds, w_preds = preds
        w_preds, w_target = TaggerTrainer.remove_padding(w_preds, w_target)
        sys.stdout.write(
            classification_report(
                d_target.data.cpu(), d_preds.cpu(), target_names=target_names[0]
            )
        )
        sys.stdout.write(
            classification_report(
                w_target.cpu(),
                w_preds.cpu(),
                target_names=target_names[1][Padding.WORD_LABEL_PAD_IDX + 1 :],
            )
        )
        frame_accuracy = JointTrainer.frame_accuracy(
            d_target.cpu(),
            w_target.cpu(),
            d_preds.cpu(),
            w_preds.cpu(),
            seq_lens.cpu(),
        )
        sys.stdout.write("\nFrame accuracy : {:.3f} \n\n".format(frame_accuracy))
        return frame_accuracy

    def test(self, model, test_iter, metadata):
        model.eval()

        preds_table = []
        frame_pairs = []
        [doc_class_names, word_class_names] = metadata["class_names"]
        word_class_names, mapping = TaggerTrainer.filter_word_labels(word_class_names)

        preds_table.append("#{0}".format(json.dumps(doc_class_names)))
        preds_table.append("#{0}".format(json.dumps(word_class_names)))
        preds_table.append(
            (
                "doc_index",
                "doc_prediction",
                "doc_label",
                "doc_scores",
                "word_predictions",
                "word_labels",
                "[word_pred:word_lab]",
                "tokens",
                "text",
                # TODO: maybe stop reporting word_chunk_match
                "word_chunk_match",
            )
        )
        all_doc_targets, all_doc_preds, all_word_targets, all_word_preds = (
            None,
            None,
            None,
            None,
        )
        for m_input, [d_targets, w_targets], context in test_iter:
            [d_out, w_out] = model(*m_input)

            if hasattr(model, "crf") and model.crf:
                w_out = model.crf.decode_crf(w_out, w_targets)
            d_preds = torch.max(d_out, 1)[1].data

            w_out = self._flatten_2d(w_out)
            w_targets = w_targets.view(-1)
            w_preds = torch.max(w_out, 1)[1].data

            w_preds, w_targets = TaggerTrainer.remove_padding(w_preds, w_targets)
            w_preds = TaggerTrainer.map_to_filtered_ids(w_preds, mapping)
            w_targets = TaggerTrainer.map_to_filtered_ids(w_targets, mapping)

            self.update_test_results(
                preds_table,
                d_out,
                w_preds,
                w_targets,
                d_preds,
                d_targets.data,
                doc_class_names,
                word_class_names,
                context[SEQ_LENS],
                context[DatasetFieldName.RAW_WORD_LABEL],
                context[DatasetFieldName.TOKEN_RANGE_PAIR],
                context[DatasetFieldName.INDEX_FIELD],
                context[DatasetFieldName.UTTERANCE_FIELD],
                frame_pairs,
            )

            # Bookkeeping to compute final metrics
            if all_word_preds is None:
                all_doc_targets = d_targets
                all_doc_preds = d_preds
                all_word_preds = w_preds
                all_word_targets = w_targets
            else:
                all_doc_targets = torch.cat((all_doc_targets, d_targets), 0)
                all_doc_preds = torch.cat((all_doc_preds, d_preds), 0)
                all_word_preds = torch.cat((all_word_preds, w_preds), 0)
                all_word_targets = torch.cat((all_word_targets, w_targets), 0)

        doc_result_table, doc_weighted_metrics = test_utils.get_all_metrics(
            all_doc_preds.cpu(), all_doc_targets.cpu(), doc_class_names
        )
        word_result_table, word_weighted_metrics = test_utils.get_all_metrics(
            all_word_preds.cpu(), all_word_targets.cpu(), word_class_names
        )

        frame_metrics = compute_all_metrics(frame_pairs)

        return (
            preds_table,
            [doc_result_table, word_result_table],
            [doc_weighted_metrics, word_weighted_metrics],
            frame_metrics,
        )

    def update_test_results(
        self,
        test_results,
        d_out,
        w_preds,
        w_targets,
        d_preds,
        d_targets,
        doc_class_names,
        word_class_names,
        seq_lens,
        raw_word_labels,
        token_range_pair,
        orig_indices,
        utterances,
        frame_pairs,
    ):
        offset = 0
        for i in range(seq_lens.size()[0]):
            w_preds_idx = w_preds[offset : offset + seq_lens[i]]
            w_target_idx = w_targets[offset : offset + seq_lens[i]]
            offset += seq_lens[i]
            w_preds_names = [word_class_names[p] for p in w_preds_idx]
            w_label_names = raw_word_labels[i]
            w_preds_names = test_utils.summarize(
                seq_lens[i], token_range_pair[i], w_preds_names
            )
            tokens = [t for t, _ in token_range_pair[i]]

            w_pred_lab = ":".join(
                [str(list(map(int, w_preds_idx))), str(list(map(int, w_target_idx)))]
            )
            test_results.append(
                (
                    orig_indices[i],
                    d_preds[i].item(),
                    d_targets[i].item(),
                    ",".join(
                        ["{0:.2f}".format(_s) for _s in F.softmax(d_out[i], 0).data]
                    ),
                    w_preds_names,
                    w_label_names,
                    w_pred_lab,
                    tokens,
                    utterances[i],
                    test_utils.count_chunk_match(w_preds_names, w_label_names),
                )
            )

            predicted_frame = JointTrainer.create_frame(
                doc_class_names, d_preds, i, w_preds_names, utterances
            )
            expected_frame = JointTrainer.create_frame(
                doc_class_names, d_targets, i, w_label_names, utterances
            )
            frame_pairs.append((predicted_frame, expected_frame))

    @staticmethod
    def create_frame(doc_class_names, doc_class_indices, i, word_names, utterances):
        frame = Node(
            label=doc_class_names[doc_class_indices[i].item()],
            span=Span(0, len(utterances[i])),
            children={
                Node(label=slot.label, span=Span(slot.start, slot.end))
                for slot in data_utils.parse_slot_string(word_names)
            },
        )
        return frame

    @staticmethod
    def frame_accuracy(d_target, w_targets, d_preds, w_preds, seq_lens):
        n_samples = len(d_target)
        assert len(seq_lens) == n_samples

        num_correct = 0
        offset = 0
        for i in range(n_samples):
            num_correct += int(
                d_preds[i] == d_target[i]
                and np.array_equal(
                    w_preds[offset : offset + seq_lens[i]],
                    w_targets[offset : offset + seq_lens[i]],
                )
            )
            offset += seq_lens[i]
        return num_correct / n_samples
