#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import time
from typing import Any, Optional, Tuple

import torch
from pytext.common.constants import BatchContext, Stage
from pytext.config import PyTextConfig
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.data.data_handler import BatchIterator
from pytext.metric_reporters import MetricReporter
from pytext.models.distributed_model import DistributedModel
from pytext.models.model import Model
from pytext.optimizer import learning_rates
from pytext.optimizer.scheduler import Scheduler
from pytext.utils import cuda_utils, time_utils


class TrainerBase(Component):
    __COMPONENT_TYPE__ = ComponentType.TRAINER


class Trainer(TrainerBase):
    """
    Base Trainer class that provide ways to
        1 Train model, compute metrics against eval set and use the metrics for
        model selection.
        2 Test trained model, compute and publish metrics against a blind test set.

    Attributes:
        epochs (int): Training epochs
        early_stop_after (int): Stop after how many epochs when the eval metric
            is not improving
        max_clip_norm (Optional[float]): Clip gradient norm if set
        report_train_metrics (bool): Whether metrics on training data should be
            computed and reported.
        target_time_limit_seconds (int): Target time limit for training in seconds. If
            the expected time to train another epoch exceeds this limit, stop training.
    """

    class Config(ConfigBase):
        # Training epochs
        epochs: int = 10
        # Stop after how many epochs when the eval metric is not improving
        early_stop_after: int = 0
        # Clip gradient norm if set
        max_clip_norm: Optional[float] = None
        # Whether metrics on training data should be computed and reported.
        report_train_metrics: bool = True
        # Target time limit for training.
        target_time_limit_seconds: int = 0

    def test(self, test_iter, model, metric_reporter: MetricReporter):
        model.eval()
        with torch.no_grad():
            test_metric = self._run_epoch(
                Stage.TEST, 1, test_iter, model, metric_reporter
            )
        return test_metric

    def train(
        self,
        train_iter: BatchIterator,
        eval_iter: BatchIterator,
        model: Model,
        metric_reporter: MetricReporter,
        train_config: PyTextConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler = None,
        rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:
        """
        Train and eval a model, the model states will be modified. This function
        iterates epochs specified in config, and for each epoch do:

            1. Train model using training data, aggregate and report training results
            2. Adjust learning rate if scheduler is specified
            3. Evaluate model using evaluation data
            4. Calculate metrics based on evaluation results and select best model

        Args:
            train_iter (BatchIterator): batch iterator of training data
            eval_iter (BatchIterator): batch iterator of evaluation data
            model (Model): model to be trained
            metric_reporter (MetricReporter): compute metric based on training
                output and report results to console, file.. etc
            train_config (PyTextConfig): training config
            optimizer (torch.optim.Optimizer): torch optimizer to be used
            scheduler (Scheduler): learning rate scheduler,
                default is None
            training_result (Optional): only meaningful for Hogwild training. default
                is None
            rank (int): only used in distributed training, the rank of the current
                training thread, evaluation will only be done in rank 0

        Returns:
            model, best_metric: the trained model together with the best metric
        """
        timer = time_utils.StageTimer()
        world_size = 1
        if cuda_utils.CUDA_ENABLED:
            model = model.cuda()
            world_size = cuda_utils.DISTRIBUTED_WORLD_SIZE
            if world_size > 1:
                device_id = torch.cuda.current_device()
                model = DistributedModel(
                    module=model,
                    device_ids=[device_id],
                    output_device=device_id,
                    broadcast_buffers=False,
                )
            timer.add_stage(stage="init_distributed_model")

        best_metric = None
        last_best_epoch = 0
        timer.add_stage(stage="pre_training")

        if scheduler:
            scheduler.prepare(train_iter, self.config.epochs)

        def training_pre_batch_callback():
            if world_size > 1:
                # replace optimizer.zero_grad() here to work with DDP
                # in cases where some parameters don't receive grads at each step
                # loss.backward will set grad for params in the computation graph
                # we can thus follow which params are left out and call .backward
                # on them manually
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad = None
            else:
                optimizer.zero_grad()

        def training_backprop(loss):
            loss.backward()
            if world_size > 1:
                # DDP fix when some parameters don't receive grads
                for p in model.parameters():
                    if p.requires_grad and p.grad is None:
                        p.backward(torch.zeros_like(p.data))
            timer.add_stage("backward")

            if scheduler:
                scheduler.step_batch()

            if self.config.max_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.max_clip_norm
                )
            else:
                grad_norm = None

            optimizer.step()
            timer.add_stage("update_grads")
            # grad_norm could be used to check grads sync in distributed training
            return grad_norm

        time_start = time.time()
        for epoch in range(1, self.config.epochs + 1):
            if self.config.target_time_limit_seconds > 0 and epoch > 1:
                time_elapsed = time.time() - time_start
                mean_epoch_time = time_elapsed / float(epoch - 1)
                expected_next_epoch_time = time_elapsed + mean_epoch_time
                if expected_next_epoch_time > self.config.target_time_limit_seconds:
                    print(
                        f"Training stopped after {epoch - 1} epochs and "
                        f"{int(time_elapsed)} seconds, due to the target max training "
                        f"time of {self.config.target_time_limit_seconds} seconds."
                    )
                    break

            print(f"Rank {rank} worker: Starting epoch #{epoch}")
            model.train()
            lrs = (str(lr) for lr in learning_rates(optimizer))
            print(f"Learning rate(s): {', '.join(lrs)}")
            self._run_epoch(
                Stage.TRAIN,
                epoch,
                train_iter,
                model,
                metric_reporter,
                pre_batch=training_pre_batch_callback,
                backprop=training_backprop,
                rank=rank,
            )
            timer.add_stage(stage=f"epoch_train")

            model.eval(Stage.EVAL)
            with torch.no_grad():
                eval_metric = self._run_epoch(
                    Stage.EVAL, epoch, eval_iter, model, metric_reporter, rank=rank
                )
            timer.add_stage(stage=f"epoch_eval")

            # Step the learning rate scheduler(s)
            if scheduler:
                assert eval_metric is not None
                scheduler.step_epoch(
                    metrics=metric_reporter.get_model_select_metric(eval_metric),
                    epoch=epoch,
                )

            # choose best model.
            if metric_reporter.compare_metric(eval_metric, best_metric):
                last_best_epoch = epoch
                best_metric = eval_metric
                best_selection_metric = metric_reporter.get_model_select_metric(
                    best_metric
                )
                # Only rank = 0 trainer saves modules.
                if train_config.save_module_checkpoints and rank == 0:
                    model.save_modules(
                        base_path=train_config.modules_save_dir, suffix=f"-ep{epoch}"
                    )

                if rank == 0:
                    print(
                        f"Rank {rank} worker: Found a better model! "
                        f"Metric for better model is {best_selection_metric}."
                    )
                    model_state = model.state_dict()
                    # save to cpu to avoid multiple model copies in gpu memory
                    if cuda_utils.CUDA_ENABLED:
                        for key, state in model_state.items():
                            model_state[key] = state.cpu()
                    best_model_state = model_state
                timer.add_stage(stage=f"epoch_save/load_module")

            if self.config.early_stop_after > 0 and (
                epoch - last_best_epoch == self.config.early_stop_after
            ):
                print(
                    f"Rank {rank} worker: Eval metric hasn't changed for "
                    + f"{self.config.early_stop_after} epochs. Stopping now."
                )
                break
            sys.stdout.flush()

        if rank == 0:
            if cuda_utils.CUDA_ENABLED:
                for key, state in best_model_state.items():
                    best_model_state[key] = state.cuda()
            model.load_state_dict(best_model_state)

        timer.report("Trainer train timer")
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
        rank=0,
    ):
        print(f"Rank {rank} worker: Running epoch #{epoch} for {stage}")
        report_metric = stage != Stage.TRAIN or self.config.report_train_metrics

        timer = time_utils.StageTimer()
        for batch_id, (inputs, targets, context) in enumerate(data_iter):
            pre_batch()
            # pass context to model to use in forward call if needed
            model.contextualize(context)
            logits = model(*inputs)
            timer.add_stage("forward")

            loss = model.get_loss(logits, targets, context)
            if BatchContext.IGNORE_LOSS in context:
                loss *= 0
            timer.add_stage("compute_loss")

            backprop(loss)
            if report_metric:
                preds, scores = model.get_pred(logits, targets, context, stage, *inputs)
                metric_reporter.add_batch_stats(
                    batch_id, preds, targets, scores, loss.item(), inputs, **context
                )
                timer.add_stage("add_metric")

        metrics = None
        if report_metric:
            metrics = metric_reporter.report_metric(
                stage, epoch, print_to_channels=(rank == 0)
            )
            timer.add_stage("report_metric")
        else:
            metric_reporter._reset()

        timer.report("Trainer epoch timer")
        return metrics
