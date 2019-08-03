#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib

import torch


"""fp16 optimizer wraps torch.optim to support mixed precision training

structure of fp16Optimizer:
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
Properties:
    inner_optimizer:
        type: Torch.optim
        contents: optimizer in pytext (eg. Adam)
                    which is initialized with fp16 params already
    param_groups:
        type: list of dictionaries where key is string and value is a list.
        contents: eg. [{'params':[]}]
    temp_fp32_params
        types: same as param_groups
        purpose: to support accumulating grads calculation
        contents: contain the temp fp32 grads from backward()
                    and will be unscaled and added to inner optimizer
    scaler:
    flags: BOOLEAN
        weights_update_needed: whether need to copy weights from master to model
        grads_update_needed: whether need to copy grads from model to master
Methods:
__init__()
zero_grad
effects: clear the grads in self.param_groups(fp16)
step()

class DynamicLossScaler:
 properties:
    init_scale: the beginning scale number
    scale_factor: the step length that we use to increase the scale
    scale_window: the upper bound of iterations among which no overflow is triggered
    is_overflow
    is_scaled: whether grads are scaled
Methods:
    check_overflow
    upscale
    unscale
    update_scale:
        effects:
        if last overflow is far from now, it's time to increase scale
        if more overflow happens than we expected, it's time to decrease the scale
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
        grad.div_(self.scale)

    def unscale_grads(self, param_groups):
        for p in generate_params(param_groups):
            self.unscale(p.grad)

    def check_overflow(self, master_params):
        self.is_overflow = False
        for p in generate_params(master_params):
            if p.grad is not None:
                cpu_sum = float(p.grad.float().sum())
                if (
                    cpu_sum == float("inf")
                    or cpu_sum == -float("inf")
                    or cpu_sum != cpu_sum
                ):
                    self.is_overflow = True
                    break

    def update_scale(self):
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
        input:
        init_optimizer(initialized already), init_scale, scale_factor, scale_window
        effects:
        initialize the optimizer, create master and loss scaling tools
        modifies:
        record the reference of model params (fp16), change the inner optimizer's
        params to fp32, initialized the scaler, state and default
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
        for p in generate_params(self.param_groups):
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def scale_loss(self, loss):
        self.grads_update_needed = True
        return self.loss_scaler.upscale(loss)

    def step(self):
        """
        effects:
        update the grads from model to master
        check whether model grads are overflow
        call inner optimizer's step()
        copy back the weights from inner optimizer to model
        """
        self._grads_from_model_to_master()
        self.loss_scaler.check_overflow(self.inner_optimizer.param_groups)
        if not self.loss_scaler.is_overflow:
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
        state_dict["overflow"] = self.loss_scaler.is_overflow
        state_dict["param_groups"] = self.param_groups
        state_dict["optimizer_state_dict"] = self.inner_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.loss_scaler.scale = state_dict["loss_scale"]
        self.loss_scaler.is_overflow = state_dict["overflow"]
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        for model_param, state_param in zip(
            generate_params(self.param_groups),
            generate_params(state_dict["param_groups"]),
        ):
            model_param.data.copy_(state_param.data)

    def finalize(self):
        return self.inner_optimizer.finalize()

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)


def initialize(
    model,
    optimizer,
    opt_level,
    init_scale=2 ** 16,
    scale_factor=2.0,
    scale_window=2000,
    memory_efficient=False,
):
    optimizer = (
        FP16Optimizer(optimizer, init_scale, scale_factor, scale_window)
        if not memory_efficient
        else PureFP16Optimizer(optimizer, init_scale, scale_factor, scale_window)
    )

    return (model.half(), optimizer)


@contextlib.contextmanager
def scale_loss(loss, optimizer, delay_unscale=False):
    yield optimizer.scale_loss(loss)


def master_params(optimizer):
    return generate_params(optimizer.inner_optimizer.param_groups)


def generate_params(param_groups):
    for group in param_groups:
        for p in group["params"]:
            yield p


"""PureFP16Optimizer
No maintenance of fp32 weights.

Internally maintain the chain:

loss.backward()          float()          step()               half()
----------------->fp16 grads------>fp32 grads------> fp32 weights -----> fp16 weights

"""


class PureFP16Optimizer(object):
    def __init__(
        self, init_optimizer, init_scale=2.0 ** 16, scale_factor=2, scale_window=2000
    ):
        """
        input:
        init_optimizer(initialized already), init_scale, scale_factor, scale_window
        effects:
        initialize this optimizer wrapper and loss scaling tools,
        initialized the scaler and state
        """
        self.inner_optimizer = init_optimizer
        self.param_groups = self.inner_optimizer.param_groups
        self.loss_scaler = DynamicLossScaler(init_scale, scale_factor, scale_window)
        self.state = self.inner_optimizer.state
        self.is_scaled = False
        print("==================Memory Efficient Optimizer===================")

    def zero_grad(self):
        for p in generate_params(self.param_groups):
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def scale_loss(self, loss):
        self.is_scaled = True
        return self.loss_scaler.upscale(loss)

    def step(self):
        """
        effects:
        if inner optimizer supports memory efficient, check overflow,
        unscale and call advanced step
        otherwise, float weights and grads, check whether grads are overflow
        if not overflow,unscale grads and call inner optimizer's step
        Otherwise, do nothing, wait to the end to call half
        half weights and grads (grads will be eliminated in zero_grad)
        """
        support = getattr(self.inner_optimizer, "supports_memory_efficient_fp16", False)

        if not support:
            self._fp16_to_fp32()

        self.loss_scaler.check_overflow(self.param_groups)
        if not self.loss_scaler.is_overflow:
            self._unscale()
            self.inner_optimizer.step()

        if not support:
            self._fp32_to_fp16()

        self.loss_scaler.update_scale()

    def _unscale(self):
        if self.is_scaled:
            self.loss_scaler.unscale_grads(self.param_groups)
            self.is_scaled = False

    def _fp16_to_fp32(self):
        for p in generate_params(self.param_groups):
            p.data = p.data.float()
            if p.grad is not None:
                p.grad.data = p.grad.data.float()

    def _fp32_to_fp16(self):
        for p in generate_params(self.param_groups):
            p.data = p.data.half()
            if p.grad is not None:
                p.grad.data = p.grad.data.half()

    def state_dict(self):
        state_dict = {}
        state_dict["loss_scale"] = self.loss_scaler.scale
        state_dict["overflow"] = self.loss_scaler.is_overflow
        state_dict["param_groups"] = self.param_groups
        state_dict["optimizer_state_dict"] = self.inner_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.loss_scaler.scale = state_dict["loss_scale"]
        self.loss_scaler.is_overflow = state_dict["overflow"]
        self.param_groups = state_dict["param_groups"]
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def finalize(self):
        return self.inner_optimizer.finalize()

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)
