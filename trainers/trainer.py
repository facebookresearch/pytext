#!/usr/bin/env python3

import copy
from typing import List, Optional

import torch
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.loss.loss import Loss
from pytext.optimizer import optimizer_step, optimizer_zero_grad, scheduler_step
from pytext.utils import cuda_utils
from pytext.utils.data_utils import Slot


class Trainer(Component):
    __COMPONENT_TYPE__ = ComponentType.TRAINER

    class Config(ConfigBase):
        # Manual random seed
        random_seed: int = 0
        # Training epochs
        epochs: int = 10
        # Stop after how many epochs when the eval metric is not improving
        early_stop_after: int = 0
        # Print the training metrics every log_interval epochs
        log_interval: int = 1
        # Evaluate the model every eval_interval epochs
        eval_interval: int = 1
        # Clip gradient norm if set
        max_clip_norm: Optional[float] = None

    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(config)

    def report(self, stage, loss, preds, seq_lens, targets, target_names):
        # Print training/eval metrics report
        # It has to be implemented by all the subclasses of Trainer
        raise NotImplementedError()

    def test(self, model, test_iter, metadata):
        raise NotImplementedError()

    def train(
        self,
        train_iter,
        eval_iter,
        model,
        optimizers: List[torch.optim.Optimizer],
        loss_fn: Loss,
        labels,
        metrics_reporter=None,
        scheduler=None,
    ):
        # TODO this var will be part of MetricReporter T33077795
        label_names = [label.vocab.itos for label in labels.values()]
        if cuda_utils.CUDA_ENABLED:
            model.cuda()

        model.train()
        best_metric = float("-inf")
        last_best_epoch = 0
        best_model = None
        train_loss_sum = 0.0
        n_batches = 0

        # TODO: This needs to be unified. Model with CRF should use CRF loss.
        self.use_crf = True if hasattr(model, "crf") and model.crf else False
        assert self.use_crf == (
            hasattr(loss_fn, "is_crf") and loss_fn.is_crf()
        ), "CRFLoss is needed for CRF models"

        for _epoch in range(1, self.config.epochs + 1):
            print("Starting epoch# {}".format(_epoch))
            if scheduler:
                scheduler_step(scheduler)
            for m_input, targets, context in train_iter:
                optimizer_zero_grad(optimizers)

                m_out = model(*m_input)
                # Flatten the words logits tensor from (N, L, D) to a 2D tensor
                # of size (N*L, D), doc logits tensor won't be affected
                # Postpone till later if using crf loss

                if not self.use_crf:
                    m_out = [self._flatten_2d(logits) for logits in m_out]
                    targets = [target.view(-1) for target in targets]

                loss = loss_fn.loss(m_out, targets, model, context)
                train_loss_sum += loss.item()
                n_batches += 1

                loss.backward()
                optimizer_step(optimizers)

            train_metric, train_loss, eval_metric, eval_loss = None, None, None, None
            if self.use_crf:
                # decode and arrange such that the decoded word has max prob
                m_out[-1] = model.crf.decode_crf(m_out[-1], targets[-1])
                m_out = [self._flatten_2d(logits) for logits in m_out]
                targets = [target.view(-1) for target in targets]

            if _epoch % self.config.log_interval == 0:
                preds = [
                    torch.max(logit, 1)[1].view(targets[i].size()).data
                    for i, logit in enumerate(m_out)
                ]
                # Report Train Loss per batch
                train_loss = train_loss_sum / float(n_batches)
                train_metric = self.report(
                    "Training-Epoch-Snapshot[{}]".format(_epoch),
                    train_loss,
                    preds,
                    context[SEQ_LENS],
                    targets,
                    label_names,
                )
                # Reset train_loss_sum and n_batches to 0 for reporting next time
                train_loss_sum = 0.0
                n_batches = 0

            if _epoch % self.config.eval_interval == 0:
                eval_metric, eval_loss = self.evaluate(
                    eval_iter, model, loss_fn, label_names
                )

                if eval_metric > best_metric:
                    last_best_epoch = _epoch
                    best_metric = eval_metric
                    print("Found a better model! Saving it...")
                    best_model = copy.deepcopy(model)

            if self.config.early_stop_after > 0 and (
                _epoch - last_best_epoch == self.config.early_stop_after
            ):
                print(
                    "Eval metric hasn't changed for {} epochs, stopping now...".format(
                        self.config.early_stop_after
                    )
                )
                break

            if (
                train_metric is not None
                and train_loss is not None
                and eval_metric is not None
                and eval_loss is not None
                and metrics_reporter is not None
            ):
                metrics_reporter(
                    _epoch, train_loss, eval_loss, train_metric, eval_metric
                )

        return best_model

    def evaluate(self, eval_iter, model, loss_fn, label_names):
        model.eval()
        all_targets, all_preds, all_seq_lengths = None, None, None
        total_loss, n_batches = 0, 0
        for m_input, targets, context in eval_iter:
            m_out = model(*m_input)
            n_batches += 1
            # The CRF loss needs to be obtained before flatenning
            if self.use_crf:
                total_loss += loss_fn.loss(m_out, targets, model, context).item()
                # The last output is word
                m_out[-1] = model.crf.decode_crf(m_out[-1], targets[-1])
                m_out = [self._flatten_2d(logits) for logits in m_out]
                targets = [target.view(-1) for target in targets]
            else:
                m_out = [self._flatten_2d(logits) for logits in m_out]
                targets = [target.view(-1) for target in targets]
                total_loss += loss_fn.loss(m_out, targets, model, context).item()

            preds = [
                torch.max(logit, 1)[1].view(targets[i].size()).data
                for i, logit in enumerate(m_out)
            ]

            if all_targets is None:
                all_targets = targets
                all_preds = preds
                all_seq_lengths = context[SEQ_LENS]
            else:
                for i, target in enumerate(targets):
                    all_targets[i] = torch.cat((all_targets[i], target), 0)
                    all_preds[i] = torch.cat((all_preds[i], preds[i]), 0)
                all_seq_lengths = torch.cat((all_seq_lengths, context[SEQ_LENS]), 0)

        model.train()
        total_loss = total_loss / float(n_batches)

        eval_acc = self.report(
            "Evaluation",
            total_loss,
            all_preds,
            all_seq_lengths,
            all_targets,
            label_names,
        )

        return eval_acc, total_loss

    def _flatten_2d(self, in_tensor):
        return in_tensor.view(-1, in_tensor.size()[-1])

    def _filter_bio_prefix(self, word_label_names):
        # Transform labels to non-BIO format
        filtered_names = []
        for c_name in word_label_names:
            to_strip = 0
            if c_name.startswith(Slot.B_LABEL_PREFIX) or c_name.startswith(
                Slot.I_LABEL_PREFIX
            ):
                to_strip = len(Slot.B_LABEL_PREFIX)
            filtered_names.append(c_name[to_strip:])
        return filtered_names
