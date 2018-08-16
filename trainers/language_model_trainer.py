#!/usr/bin/env python3

import copy
import math
import sys
from typing import List

import torch
from pytext.common.registry import TRAINER, component
from pytext.config.pytext_config import ConfigBase
from pytext.loss.loss import Loss
from pytext.optimizers import optimizer_step, optimizer_zero_grad
from pytext.utils import cuda_utils

from .trainer import Trainer, TrainerConfig


class LMTrainerConfig(ConfigBase, TrainerConfig):
    pass


@component(TRAINER, config_cls=LMTrainerConfig)
class LanguageModelTrainer(Trainer):
    def report(self, stage, loss):
        sys.stdout.write("{} - loss: {:.6f}\n".format(stage, loss))
        perplexity = math.exp(loss)
        sys.stdout.write("{} - perplexity: {:.6f}\n".format(stage, perplexity))
        return perplexity

    def train(
        self,
        train_iter,
        eval_iter,
        model,
        optimizers: List[torch.optim.Optimizer],
        loss_fn: Loss,
        class_names,
        metrics_reporter=None,
    ):
        if cuda_utils.CUDA_ENABLED:
            model.cuda()

        model.train()
        # The metric used is perplexity. Lower the perplexity, better the model
        best_perplexity = float("inf")
        last_best_epoch = 0
        best_model = None
        train_loss_sum = 0.0
        n_batches = 0

        for _epoch in range(1, self.params.epochs + 1):
            print("Starting epoch# {}".format(_epoch))
            for m_input, targets, context in train_iter:
                optimizer_zero_grad(optimizers)
                m_out = model(*m_input)
                # Flatten the logits tensor from (N, L, D) to a 2D tensor
                # of size (N*L, D)˜
                m_out = [self._flatten_2d(logits) for logits in m_out]
                targets = [target.view(-1) for target in targets]

                loss = loss_fn.loss(m_out, targets, model, context)
                train_loss_sum += loss.item()
                n_batches += 1

                loss.backward()
                optimizer_step(optimizers)

            # The metric here is perplexity
            train_perplexity, train_loss, eval_perplexity, eval_loss = (
                None,
                None,
                None,
                None,
            )

            if _epoch % self.params.log_interval == 0:
                # Report Train Loss per batch
                train_loss = train_loss_sum / float(n_batches)
                train_perplexity = self.report(
                    "Training-Epoch-Snapshot[{}]".format(_epoch), train_loss
                )
                # Reset train_loss_sum and n_batches to 0 for reporting next time
                train_loss_sum = 0.0
                n_batches = 0

            if _epoch % self.params.eval_interval == 0:
                eval_perplexity, eval_loss = self.evaluate(eval_iter, model, loss_fn)

                # Lower perplexity implies a better model
                if eval_perplexity < best_perplexity:
                    last_best_epoch = _epoch
                    best_perplexity = eval_perplexity
                    print("Found a better model! Saving it...")
                    best_model = copy.deepcopy(model)

            if self.params.early_stop_after > 0 and (
                _epoch - last_best_epoch == self.params.early_stop_after
            ):
                print(
                    "Eval metric hasn't changed for {} epochs, stopping now...".format(
                        self.params.early_stop_after
                    )
                )
                break

            if (
                train_perplexity is not None
                and train_loss is not None
                and eval_perplexity is not None
                and eval_loss is not None
                and metrics_reporter is not None
            ):
                metrics_reporter(
                    _epoch, train_loss, eval_loss, train_perplexity, eval_perplexity
                )

        return best_model

    def evaluate(self, eval_iter, model, loss_fn):
        model.eval()
        all_targets = None
        total_loss, n_batches = 0, 0
        for m_input, targets, context in eval_iter:
            m_out = model(*m_input)
            n_batches += 1

            m_out = [self._flatten_2d(logits) for logits in m_out]
            targets = [target.view(-1) for target in targets]
            total_loss += loss_fn.loss(m_out, targets, model, context).item()

            if all_targets is None:
                all_targets = targets
            else:
                for i, target in enumerate(targets):
                    all_targets[i] = torch.cat((all_targets[i], target), 0)

        model.train()
        average_batch_loss = total_loss / float(n_batches)

        eval_perplexity = self.report("Evaluation", average_batch_loss)

        return eval_perplexity, average_batch_loss
