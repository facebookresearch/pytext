#!/usr/bin/env python3

import json
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName, Padding
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.metrics import FramePredictionPair, Node, Span, compute_all_metrics
from pytext.utils.data_utils import parse_slot_string
from pytext.utils.test_utils import summarize
from sklearn.metrics import classification_report

from .tagger_trainer import TaggerTrainer
from .trainer import Trainer


class JointTrainer(Trainer):
    # TODO: (wenfangxu) T32687556 define unified interfaces for report() and test()
    def report(self, stage, loss, preds, seq_lens, targets, target_names):
        d_targets = []
        w_targets = []
        for [d, w] in targets:
            d_targets.append(d)
            w_targets.append(w.view(-1))
        d_target = torch.cat(d_targets, 0)
        w_target = torch.cat(w_targets, 0)

        d_preds = []
        w_preds = []
        for [d, w] in preds:
            d_preds.append(d)
            w_preds.append(w.view(-1))
        d_pred = torch.cat(d_preds, 0)
        w_pred = torch.cat(w_preds, 0)

        w_pred, w_target = TaggerTrainer.remove_padding(w_pred, w_target)
        sys.stdout.write(
            classification_report(
                d_target.data.cpu(), d_pred.cpu(), target_names=target_names[0]
            )
        )
        sys.stdout.write(
            classification_report(
                w_target.cpu(),
                w_pred.cpu(),
                target_names=target_names[1][Padding.WORD_LABEL_PAD_IDX + 1 :],
            )
        )
        frame_accuracy = JointTrainer.frame_accuracy(
            d_target.cpu(), w_target.cpu(), d_pred.cpu(), w_pred.cpu(), seq_lens.cpu()
        )
        sys.stdout.write("\nFrame accuracy : {:.3f} \n\n".format(frame_accuracy))
        return frame_accuracy

    def test(self, model, test_iter, metadata):
        model.eval()

        preds_table = []
        frame_pairs: List[FramePredictionPair] = []

        [doc_labe_meta, word_label_meta] = metadata.labels.values()
        [doc_label_names, word_label_names] = [
            doc_labe_meta.vocab.itos,
            word_label_meta.vocab.itos,
        ]
        word_label_names, mapping = TaggerTrainer.filter_word_labels(word_label_names)

        preds_table.append("#{0}".format(json.dumps(doc_label_names)))
        preds_table.append("#{0}".format(json.dumps(word_label_names)))
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
            preds, scores = model.get_pred(
                [d_out, w_out], context
            )

            [d_preds, w_preds] = preds

            w_preds, w_targets = TaggerTrainer.remove_padding(
                w_preds.view(-1), w_targets.view(-1)
            )
            w_preds = TaggerTrainer.map_to_filtered_ids(w_preds, mapping)
            w_targets = TaggerTrainer.map_to_filtered_ids(w_targets, mapping)

            self.update_test_results(
                preds_table,
                d_out,
                w_preds,
                w_targets,
                d_preds,
                d_targets.data,
                doc_label_names,
                word_label_names,
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

        frame_metrics = compute_all_metrics(frame_pairs, frame_accuracy=True)

        return preds_table, frame_metrics

    def update_test_results(
        self,
        test_results,
        d_out,
        w_preds,
        w_targets,
        d_preds,
        d_targets,
        doc_label_names,
        word_label_names,
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
            w_preds_names = [word_label_names[p] for p in w_preds_idx]
            w_label_names = raw_word_labels[i]
            w_preds_names = summarize(seq_lens[i], token_range_pair[i], w_preds_names)
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
                )
            )

            predicted_frame = JointTrainer.create_frame(
                doc_label_names, d_preds, i, w_preds_names, utterances
            )
            expected_frame = JointTrainer.create_frame(
                doc_label_names, d_targets, i, w_label_names, utterances
            )
            frame_pairs.append(FramePredictionPair(predicted_frame, expected_frame))

    @staticmethod
    def create_frame(doc_label_names, doc_class_indices, i, word_names, utterances):
        frame = Node(
            label=doc_label_names[doc_class_indices[i].item()],
            span=Span(0, len(utterances[i])),
            children={
                Node(label=slot.label, span=Span(slot.start, slot.end))
                for slot in parse_slot_string(word_names)
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
