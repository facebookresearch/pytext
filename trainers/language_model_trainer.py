#!/usr/bin/env python3

import copy
import math
import sys
from typing import List

import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.optimizer import optimizer_step, optimizer_zero_grad, scheduler_step
from pytext.utils import cuda_utils

from .trainer import Trainer


class LanguageModelTrainer(Trainer):
    def calculate_perplexity(self, loss):
        return math.exp(loss)

    def report(self, stage, loss):
        sys.stdout.write("{} - loss: {:.6f}\n".format(stage, loss))
        perplexity = self.calculate_perplexity(loss)
        sys.stdout.write("{} - perplexity: {:.6f}\n".format(stage, perplexity))
        return perplexity

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
        if cuda_utils.CUDA_ENABLED:
            model.cuda()

        model.train()
        # The metric used is perplexity. Lower the perplexity, better the model
        best_perplexity = float("inf")
        last_best_epoch = 0
        best_model = None
        train_loss_sum = 0.0
        n_words = 0

        for _epoch in range(1, self.config.epochs + 1):
            print("Starting epoch# {}".format(_epoch))
            if scheduler:
                scheduler_step(scheduler)
            for m_input, targets, context in train_iter:
                optimizer_zero_grad(optimizers)

                logit = model(*m_input)
                loss = model.get_loss(logit, targets, context)
                num_words_in_batch = torch.sum(m_input[1]).item()
                train_loss_sum += loss.item() * num_words_in_batch
                n_words += num_words_in_batch

                loss.backward()
                if self.config.max_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_clip_norm)
                optimizer_step(optimizers)

            # The metric here is perplexity
            train_perplexity, train_loss, eval_perplexity, eval_loss = (
                None,
                None,
                None,
                None,
            )

            if _epoch % self.config.log_interval == 0:
                # Report Train Loss per batch
                train_loss = train_loss_sum / float(n_words)
                train_perplexity = self.report(
                    "Training-Epoch-Snapshot[{}]".format(_epoch), train_loss
                )
                # Reset train_loss_sum and n_batches to 0 for reporting next time
                train_loss_sum = 0.0
                n_words = 0

            if _epoch % self.config.eval_interval == 0:
                eval_perplexity, eval_loss = self.evaluate(eval_iter, model)

                # Lower perplexity implies a better model
                if eval_perplexity < best_perplexity:
                    last_best_epoch = _epoch
                    best_perplexity = eval_perplexity
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

    def evaluate(self, eval_iter, model):
        model.eval()
        total_loss, n_words = 0, 0
        for m_input, targets, context in eval_iter:
            logit = model(*m_input)
            loss = model.get_loss(logit, targets, context)

            num_words_in_batch = torch.sum(m_input[1]).item()
            n_words += num_words_in_batch
            total_loss += loss.item() * num_words_in_batch

        model.train()
        loss_per_word = total_loss / float(n_words)

        eval_perplexity = self.report("Evaluation", loss_per_word)

        return eval_perplexity, loss_per_word

    def test(self, model, test_iter, metadata):
        model.eval()
        token_meta = metadata.features[DatasetFieldName.TEXT_FIELD]

        # Write header lines
        preds_table = []
        preds_table.append(("text", "perplexity"))
        total_loss = 0.0
        n_words = 0
        for m_input, targets, context in test_iter:
            logits = model(*m_input)
            # m_out dim: (bsize x seq_len x vocab)
            # Reshape m_out to (bsize x vocab x seq_len) for cross_entropy_loss
            logits = logits.transpose(1, 2)

            # While calculating loss/perplexity, we would like to mask out the
            # loss from the padding token
            weight = torch.ones(token_meta.vocab_size)
            # Mask the loss from padding token
            weight[token_meta.pad_token_idx] = 0

            # Set correct device for the weight tesnor
            if next(model.parameters()).is_cuda:
                weight = weight.cuda()

            # loss dim: (bsize x seq_len)
            loss = F.cross_entropy(logits, targets, reduce=False, weight=weight)

            num_words_in_batch = torch.sum(m_input[1]).item()
            n_words += num_words_in_batch
            total_loss += torch.sum(loss).item()

            # m_input[1] s the length of each sequence
            # sequence_loss is the loss per word for each sequence in the batch
            # sequence_loss dim: (bsize,)
            sequence_loss = loss.sum(1) / m_input[1].float()
            if DatasetFieldName.TOKEN_RANGE_PAIR in context:
                self.update_test_results(
                    preds_table, sequence_loss,
                    context[DatasetFieldName.TOKEN_RANGE_PAIR]
                )

        # Return  perplexity for every utterance and average perplexity
        # for all utterances, for now. There are no Frame metrics here
        # # TODO: Figure out a better way to abstract the metrics reporting
        return preds_table, self.calculate_perplexity(total_loss / float(n_words))

    def update_test_results(self, preds_table, sequence_loss, token_range_pair):
        for i in range(len(token_range_pair)):
            tokens = [t for t, _ in token_range_pair[i]]
            preds_table.append(
                (" ".join(tokens).strip(), self.calculate_perplexity(sequence_loss[i]))
            )
