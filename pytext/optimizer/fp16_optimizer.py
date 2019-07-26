#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib

import torch


"""fp16 optimizer wraps torch.optim to support mixed precision training

structure of fp16Optimzier:
                                      property
        fp16_optimizer.param_groups ----------> inner_optimizer.param_groups
                        |                                   |
                 ___ __ |__ __ __                  __ __ __ | __ __ __
                |     fp16       | after backward  |      fp32        |
  zero_grad ----|->   grads    --|-----------------|-->   grads    <--|-- check overflow
       loss --->|    weights   <-|-----------------|--   weights      |
      model --->|_ __ __ __ __ __|  after step     |__ __ __ __ __ __ |

usage:
1    optim.zero_grad()
2    for i in range(N):
3        model.forward()  ---- fp16 weights
4        [pre_process()]  ---- fp16 grads upscale
5        optim.backward() ---- upscaled fp16 grads
6        [post_process()] ---- downscale and float to fp32 grads
7    optim.step()         ---- fp32 weights and grads

class FP16_Optimizer:
= Properties:
    - inner_optimizer:
        = type: Torch.optim
        = contents: optimizer in pytext (eg. Adam)
                    which is initialized with fp16 params already
    - param_groups:
        = type: list of dictionaries where key is string and value is a list.
        = contents: eg. [{'params':[]}]
    - temp_fp32_params
        = types: same as param_groups
        = purpose: to support accumulating grads calculation
        = contents: contain the temp fp32 grads from backward()
                    and will be unscaled and added to inner optimizer
    - scaler:
    - flags: BOOLEAN
        = weights_update_needed: whether need to copy weights from master to model
        = grads_update_needed: whether need to copy grads from model to master
= Methods:
    - __init__()
    - zero_grad
        = effects: clear the grads in self.param_groups(fp16)
    - backward()
    - post_process()
    - step(loss)

class DynamicLossScaler:
= properties:
    - init_scale: the beginning scale number
    - scale_factor: the step length that we use to increase the scale
    - scale_window: the upper bound of iterations among which no overflow is triggered
    - tolerance: the upper bound of the frequency that overflow happens
    - threshold: the minimum value of the scale
    - is_overflow
    - is_scaled: whether grads are scaled
= Methods:
    - check_overflow
    - upscale
    - unscale
    - update_scale
"""


class DynamicLossScaler(object):
    def __init__(self, init_scale, scale_factor, scale_window):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._iter = 0
        self._last_overflow_iter = 0
        self.is_overflow = False

    def upscale(self, loss):
        return loss.float() * self.scale

    def unscale(self, grad):
        return grad.div_(self.scale)

    def check_overflow_step(self, model_params):
        for p in generate_params(model_params):
            if p.grad is not None:
                cpu_sum = float(p.grad.float().sum())
                if (
                    cpu_sum == float("inf")
                    or cpu_sum == -float("inf")
                    or cpu_sum != cpu_sum
                ):
                    return True
        return False

    def update_scale(self):
        """
        = effects:
            - if last overflow is far from now, it's time to increase scale
            - if more overflow happens than we expected, it's time to decrease the scale
        """
        self._iter += 1
        if self.is_overflow:
            self._last_overflow_iter = self._iter
            self.scale = max(self.scale / self.scale_factor, 1)
            print(
                "overflow happens, skip step, new loss scale is {}".format(self.scale)
            )
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.scale *= self.scale_factor


class FP16Optimizer(object):
    def __init__(self, init_optimizer, init_scale, scale_factor, scale_window):
        """
        = input: init_optimizer(initialized already), init_scale, scale_factor,
                    scale_window, tolerance, threshold
        = effects: initialize the optimizer and create master and loss scaling tools
        = modifies:
            - record the reference of model params (fp16)
            - change the inner optimizer's params to fp32 with
              torch.optim inner method
            - initialized the scaler
            - initialized state, default
        """
        self.inner_optimizer = init_optimizer
        self.param_groups = []
        for group in self.inner_optimizer.param_groups:
            fp16_group = {}
            for key, value in group.items():
                if key == "params":
                    fp16_param = []
                    for j, p in enumerate(value):
                        fp16_param.append(p)
                        master_p = p.detach().clone().float()
                        master_p.requires_grad_(True)
                        group["params"][j] = master_p
                        # change the state map:
                        if p in self.inner_optimizer.state:
                            self.inner_optimizer.state[
                                master_p
                            ] = self.inner_optimizer.state.pop(p)

                    fp16_group["params"] = fp16_param
                else:
                    fp16_group[key] = value
            self.param_groups.append(fp16_group)
        self.loss_scaler = DynamicLossScaler(init_scale, scale_factor, scale_window)
        self.state = self.inner_optimizer.state
        self.weights_update_needed = False
        self.grads_update_needed = False

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def scale_loss(self, loss):
        self.grads_update_needed = True
        return self.loss_scaler.upscale(loss)

    def step(self):
        """
        = input: closure
        = effects:
            - check model grads whether are overflow
            - update the grads from model to master
            - call inner optimizer's step
            - copy back the weights from inner optimizer to model
        """
        self.loss_scaler.is_overflow = self.loss_scaler.check_overflow_step(
            self.param_groups
        )
        if not self.loss_scaler.is_overflow:
            self._grads_from_model_to_master()
            self.inner_optimizer.step()
            self.weights_update_needed = True
            self._weights_from_master_to_model()
        self.loss_scaler.update_scale()

    def _grads_from_model_to_master(self):
        if self.grads_update_needed:
            for model_param, master_param in zip(
                generate_params(self.param_groups),
                generate_params(self.inner_optimizer.param_groups),
            ):
                if master_param.grad is None:
                    master_param.grad = torch.empty_like(master_param)
                master_param.grad.copy_(model_param.grad)
                self.loss_scaler.unscale(master_param.grad)
            self.grads_update_needed = False

    def _weights_from_master_to_model(self):
        if self.weights_update_needed:
            for model_param, master_param in zip(
                generate_params(self.param_groups),
                generate_params(self.inner_optimizer.param_groups),
            ):
                model_param.data.copy_(master_param.data)
            self.weights_update_needed = False

    def state_dict(self):
        state_dict = {}
        state_dict["loss_scale"] = self.loss_scaler.scale
        state_dict["overflow"] = self.loss_scaler.overflow
        state_dict["param_groups"] = self.param_groups
        state_dict["optimizer_state_dict"] = self.inner_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.loss_scaler.scale = state_dict["loss_scale"]
        self.loss_scaler.overflow = state_dict["overflow"]
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        for model_param, state_param in zip(
            generate_params(self.param_groups),
            generate_params(state_dict["param_groups"]),
        ):
            model_param.data.copy_(state_param.data)

    def finalize(self):
        return self.inner_optimizer.finalize()

    def __getstate__(self):
        raise RuntimeError("FP16_Optimizer should be serialized using state_dict().")

    def __setstate__(self, state):
        raise RuntimeError(
            "FP16_Optimizer should be deserialized using load_state_dict()."
        )

    def _get_loss_scale(self):
        return self.loss_scaler.scale

    def _set_loss_scale(self, value):
        self.loss_scaler.scale = value

    def _get_state(self):
        return self.state

    def _set_state(self, value):
        self.state = value

    def _get_param_groups(self):
        return self.param_groups

    def _set_param_groups(self, value):
        self.param_groups = value


def initialize(
    model, optimizer, opt_level, init_scale=2 ** 16, scale_factor=2.0, scale_window=2000
):
    return (
        model.half(),
        FP16Optimizer(optimizer, init_scale, scale_factor, scale_window),
    )


@contextlib.contextmanager
def scale_loss(loss, optimizer, delay_unscale=False):
    yield optimizer.scale_loss(loss)


def master_params(optimizer):
    return generate_params(optimizer.inner_optimizer.param_groups)


def generate_params(param_groups):
    for group in param_groups:
        for p in group["params"]:
            yield p
