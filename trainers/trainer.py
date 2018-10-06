#!/usr/bin/env python3

import copy
from typing import List, Optional

import torch
from pytext.common.constants import Stage
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.metric_reporters import MetricReporter
from pytext.optimizer import optimizer_step, optimizer_zero_grad, scheduler_step
from pytext.utils import cuda_utils


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

    def test(self, test_iter, model, metric_reporter: MetricReporter):
        model.eval()
        return self._run_epoch(Stage.TEST, 1, test_iter, model, metric_reporter)

    def train(
        self,
        train_iter,
        eval_iter,
        model,
        metric_reporter: MetricReporter,
        optimizers: List[torch.optim.Optimizer],
        scheduler=None,
    ):
        if cuda_utils.CUDA_ENABLED:
            model.cuda()

        best_metric = None
        last_best_epoch = 0
        best_model = None

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            print("Starting epoch# {}".format(epoch))
            self._run_epoch(
                Stage.TRAIN,
                epoch,
                train_iter,
                model,
                metric_reporter,
                optimizers,
                scheduler,
            )

            model.eval()
            eval_metric = self._run_epoch(
                Stage.EVAL, epoch, eval_iter, model, metric_reporter
            )

            # choose best model
            if metric_reporter.compare_metric(eval_metric, best_metric):
                last_best_epoch = epoch
                best_metric = eval_metric
                print("Found a better model! Saving it...")
                best_model = copy.deepcopy(model)

            if self.config.early_stop_after > 0 and (
                epoch - last_best_epoch == self.config.early_stop_after
            ):
                print(
                    "Eval metric hasn't changed for {} epochs, stopping now.".format(
                        self.config.early_stop_after
                    )
                )
                break

        return best_model, best_metric

    def _run_epoch(
        self,
        stage,
        epoch,
        data_iter,
        model,
        metric_reporter,
        optimizers=None,
        scheduler=None,
    ):
        for n_batches, (m_input, targets, context) in enumerate(data_iter):
            if model.training:
                if scheduler:
                    scheduler_step(scheduler)
                optimizer_zero_grad(optimizers)
            logits = model(*m_input)
            loss = model.get_loss(logits, targets, context)
            if model.training:
                loss.backward()
                if self.config.max_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_clip_norm
                    )
                optimizer_step(optimizers)
            preds, scores = model.get_pred(logits, targets, context)
            metric_reporter.add_batch_stats(
                n_batches, preds, targets, scores, loss.item(), **context
            )
        metrics = metric_reporter.report_metric(stage, epoch)
        return metrics
