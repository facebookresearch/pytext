#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from contextlib import AbstractContextManager
from enum import Enum
from typing import Dict

import torch
import torch.nn.functional as F
from pytext.common.constants import Stage
from pytext.config import ConfigBase
from pytext.utils.precision import maybe_float


class R3FNoiseType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"


def build_noise_sampler(noise_type: R3FNoiseType, eps: float):
    """
    Given a `noise_type` (`R3FNoiseType`): builds a `torch.distribution`
    capable of generating noise within the passed in `eps` (`float`).
    """
    if noise_type == R3FNoiseType.UNIFORM:
        return torch.distributions.uniform.Uniform(low=-eps, high=eps)
    elif noise_type == R3FNoiseType.NORMAL:
        return torch.distributions.normal.Normal(loc=0.0, scale=eps)
    else:
        raise Exception(f"Unknown noise type: {noise_type}")


def compute_symmetric_kl(noised_logits, input_logits):
    """
    Computes symmetric KL loss by taking the KL for both the input logits
    and the noised logits and comparing the two
    """
    return F.kl_div(
        F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
        F.softmax(input_logits, dim=-1, dtype=torch.float32),
        None,
        None,
        "sum",
    ) + F.kl_div(
        F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
        F.softmax(noised_logits, dim=-1, dtype=torch.float32),
        None,
        None,
        "sum",
    )  # / noised_logits.size(0)


class R3FConfigOptions(ConfigBase):
    """
    Configuration options for models using R3F
    """

    # for MTL purposes different lambda per loss
    r3f_lambda_by_loss: Dict[str, float] = {}
    r3f_default_lambda: float = 0.5
    eps: float = 1e-5
    noise_type: R3FNoiseType = R3FNoiseType.UNIFORM


class R3FNoiseContextManager(AbstractContextManager):
    """
    Context manager that adds a forward hook to the embedding module,
    to insert noise into the model and detatch embedding when doing
    this pass
    """

    def __init__(self, context):
        self.encoder_hook = None
        self.decoder_hook = None
        self.context = context
        self.hook = self.context.get_embedding_module().register_forward_hook(
            self._hook_implementation
        )

    def __enter__(self):
        return self.context

    def __exit__(self, type, value, traceback):
        self.hook.remove()
        self.hook = None

    def _hook_implementation(self, module, input, output):
        noise = self.context.noise_sampler.sample(sample_shape=output.shape).to(output)
        return output.clone().detach() + noise


class R3FPyTextMixin(object):
    """
    Mixin class for applying the R3F method, to apply R3F with any model
    inherit the class and implement the abstract functions.

    For more details: https://arxiv.org/abs/2008.03156
    """

    def __init__(self, config: R3FConfigOptions):
        self.r3f_lambda_by_loss = config.r3f_lambda_by_loss
        self.r3f_default_lambda = config.r3f_default_lambda
        self.r3f_eps = config.eps
        self.noise_sampler = build_noise_sampler(config.noise_type, self.r3f_eps)

    def get_embedding_module(self, *args, **kwargs):
        """
        Given the core model outputs, this returns the embedding module that is used
        for the R3F loss, in particular noise will be injected to this module.
        """
        raise NotImplementedError()

    def forward_with_noise(self, *args, **kwargs):
        with R3FNoiseContextManager(self):
            return self.original_forward(*args, **kwargs)

    def original_forward(self, *args, **kwargs):
        """
        Runs the traditional forward of this model
        """
        raise NotImplementedError()

    def get_sample_size(self, model_inputs, targets):
        """
        Gets the sample size of the model that is used as a regularization
        factor to the model itself
        """
        raise NotImplementedError()

    def get_r3f_model_output(self, model_output):
        """
        Extracts the output from the model.forward() call that is used for the
        r3f loss term
        """
        return model_output

    def forward(self, *args, use_r3f: bool = False, **kwargs):
        if use_r3f:
            # forward with the normal model
            model_output = self.original_forward(
                *args,
                **kwargs,
            )

            # compute noised model outputs
            noise_model_outputs = self.forward_with_noise(
                *args,
                **kwargs,
            )

            return model_output, noise_model_outputs
        else:
            return self.original_forward(*args, **kwargs)

    def get_r3f_loss_terms(
        self, model_outputs, noise_model_outputs, sample_size: int
    ) -> torch.Tensor:
        """
        Computes the auxillary loss for R3F, in particular computes a symmetric
        KL divergence between the result from the input embedding and the noise
        input embedding.
        """

        label_symm_kl = compute_symmetric_kl(
            self.get_r3f_model_output(noise_model_outputs),
            self.get_r3f_model_output(model_outputs),
        )

        label_symm_kl = label_symm_kl  # * sample_size

        return (
            self.r3f_lambda_by_loss.get("label", self.r3f_default_lambda)
            * label_symm_kl
        )

    @classmethod
    def train_batch(cls, model, batch, state=None):
        """
        Runs training over a batch with the R3F method, training will use R3F
        while eval and test do not.
        """

        # Forward pass through the network.
        model_inputs = model.arrange_model_inputs(batch)
        model_context = model.arrange_model_context(batch)
        targets = model.arrange_targets(batch)

        sample_size = model.get_sample_size(model_inputs=model_inputs, targets=targets)

        # get embedding
        r3f_loss_term = torch.tensor(0)
        if state and state.stage == Stage.TRAIN:
            # during training run R3F forward calls
            model_outputs, noise_model_outputs = model(*model_inputs, use_r3f=True)

            r3f_loss_term = model.get_r3f_loss_terms(
                model_outputs, noise_model_outputs, sample_size=sample_size
            )
        else:
            # during eval and test don't run R3F forward
            model_outputs = model(*model_inputs, use_r3f=False)

        # Add stage to context.
        if state:
            if model_context is None:
                model_context = {"stage": state.stage, "epoch": state.epoch}
            else:
                model_context["stage"] = state.stage
                model_context["epoch"] = state.epoch

        # Compute loss and predictions.
        loss = maybe_float(model.get_loss(model_outputs, targets, model_context))

        # add R3F loss term
        loss = loss + r3f_loss_term.to(loss.device)

        predictions, scores = model.get_pred(model_outputs, context=model_context)

        # Pack results and return them.
        metric_data = (predictions, targets, scores, loss, model_inputs)
        return loss, metric_data
