#!/usr/bin/env python3

import copy
from typing import List, Optional

import torch
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.data.joint_data_handler import SEQ_LENS
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
        labels,
        metrics_reporter=None,
        scheduler=None,
    ):
        # TODO this var will be part of MetricReporter T33077795
        label_names = [label.vocab.itos for label in labels.values()]
        if cuda_utils.CUDA_ENABLED:
            model.cuda()

        best_metric = float("-inf")
        last_bestepoch = 0
        best_model = None

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            print("Starting epoch# {}".format(epoch))
            train_metric, train_loss = self._run_epoch(
                train_iter, model, label_names, optimizers, scheduler
            )

            model.eval()
            eval_metric, eval_loss = self._run_epoch(eval_iter, model, label_names)

            # choose best model
            if eval_metric > best_metric:
                last_bestepoch = epoch
                best_metric = eval_metric
                print("Found a better model! Saving it...")
                best_model = copy.deepcopy(model)

            if self.config.early_stop_after > 0 and (
                epoch - last_bestepoch == self.config.early_stop_after
            ):
                print(
                    "Eval metric hasn't changed for {} epochs, stopping now.".format(
                        self.config.early_stop_after
                    )
                )
                break

            if metrics_reporter:
                metrics_reporter(
                    epoch, train_loss, eval_loss, train_metric, eval_metric
                )

        return best_model

    def _run_epoch(
        self, data_iter, model, label_names, optimizers=None, scheduler=None
    ):
        all_targets, all_preds, all_seq_lengths = [], [], []
        total_loss, n_batches = 0, 0
        for m_input, targets, context in data_iter:
            if model.training:
                if scheduler:
                    scheduler_step(scheduler)
                optimizer_zero_grad(optimizers)
            # TODO will refactor
            is_joint_model = len(targets) > 1
            if not is_joint_model:
                [targets] = targets
            logit = model(*m_input)
            loss = model.get_loss(logit, targets, context)
            total_loss += loss.item()
            if model.training:
                loss.backward()
                optimizer_step(optimizers)
            preds, scores = model.get_pred(logit, context)
            n_batches += 1
            all_targets.append(targets)
            all_preds.append(preds)
            all_seq_lengths.append(context[SEQ_LENS])
        all_seq_lengths = torch.cat(all_seq_lengths, 0)

        total_loss = total_loss / float(n_batches)

        metrics = self.report(
            "Evaluation",
            total_loss,
            all_preds,
            all_seq_lengths,
            all_targets,
            label_names,
        )
        return metrics, total_loss

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
