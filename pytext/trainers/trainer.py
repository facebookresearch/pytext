#!/usr/bin/env python3

import copy
from typing import List, Optional

import torch
from pytext.common.constants import BatchContext, Stage
from pytext.config import PyTextConfig
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.metric_reporters import MetricReporter
from pytext.models.distributed_model import DistributedModel
from pytext.optimizer import optimizer_step, optimizer_zero_grad
from pytext.utils import cuda_utils


class TrainerBase(Component):
    __COMPONENT_TYPE__ = ComponentType.TRAINER


def learning_rates(optimizers):
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            yield param_group["lr"]


class Trainer(TrainerBase):
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
        train_config: PyTextConfig,
        optimizers: List[torch.optim.Optimizer],
        scheduler=None,
    ):
        if cuda_utils.CUDA_ENABLED:
            model = model.cuda()
            if cuda_utils.DISTRIBUTED_WORLD_SIZE > 1:
                device_id = torch.cuda.current_device()
                model = DistributedModel(
                    module=model,
                    device_ids=[device_id],
                    output_device=device_id,
                    broadcast_buffers=False,
                )

        best_metric = None
        last_best_epoch = 0
        best_model_state = None

        def training_pre_batch_callback():
            optimizer_zero_grad(optimizers)

        def training_backprop(loss):
            loss.backward()
            if self.config.max_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.max_clip_norm
                )
            optimizer_step(optimizers)

        for epoch in range(1, self.config.epochs + 1):
            print("Starting epoch #{}".format(epoch))
            model.train()
            lrs = (str(lr) for lr in learning_rates(optimizers))
            print(f"Learning rate(s): {', '.join(lrs)}")

            self._run_epoch(
                Stage.TRAIN,
                epoch,
                train_iter,
                model,
                metric_reporter,
                pre_batch=training_pre_batch_callback,
                backprop=training_backprop,
            )
            model.eval()
            eval_metric = self._run_epoch(
                Stage.EVAL, epoch, eval_iter, model, metric_reporter
            )

            # Step the learning rate scheduler(s)
            if scheduler:
                scheduler.step(
                    metrics=metric_reporter.get_model_select_metric(eval_metric),
                    epoch=epoch,
                )

            # choose best model
            if metric_reporter.compare_metric(eval_metric, best_metric):
                last_best_epoch = epoch
                best_metric = eval_metric
                print("Found a better model! Saving it...")
                if train_config.save_module_checkpoints:
                    model.save_modules(
                        base_path=train_config.modules_save_dir, suffix=f"-ep{epoch}"
                    )
                best_model_state = copy.deepcopy(model.state_dict())

            if self.config.early_stop_after > 0 and (
                epoch - last_best_epoch == self.config.early_stop_after
            ):
                print(
                    "Eval metric hasn't changed for "
                    "{self.config.early_stop_after} epochs, stopping now."
                )
                break

        model.load_state_dict(best_model_state)
        return model, best_metric

    def _run_epoch(
        self,
        stage,
        epoch,
        data_iter,
        model,
        metric_reporter,
        pre_batch=lambda: None,
        backprop=lambda loss: None,
    ):
        for batch, (inputs, targets, context) in enumerate(data_iter):
            pre_batch()
            logits = model(*inputs)
            loss = model.get_loss(logits, targets, context)
            if BatchContext.IGNORE_LOSS in context:
                loss *= 0
            backprop(loss)
            preds, scores = model.get_pred(logits, targets, context, stage, *inputs)
            metric_reporter.add_batch_stats(
                batch, preds, targets, scores, loss.item(), inputs, **context
            )
        return metric_reporter.report_metric(stage, epoch)
