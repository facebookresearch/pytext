#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
from sys import stderr
from typing import Optional

import torch
from fairseq.optim.fp16_optimizer import DynamicLossScaler as Fairseq_DynamicLossScaler
from pytext.config.component import create_optimizer
from pytext.optimizer.optimizers import Optimizer
from pytext.utils import cuda, precision


_APEX_DISABLED = False
try:
    from apex import amp
except ImportError:
    print("Install apex from https://github.com/NVIDIA/apex/.", file=stderr)
    _APEX_DISABLED = True
except AttributeError as e:
    print(f"Fail to import apex: {e}", file=stderr)
    _APEX_DISABLED = True


try:
    from fairseq.optim.fp16_optimizer import (
        _FP16OptimizerMixin as Fairseq_FP16OptimizerMixin,
    )
except ImportError:
    # TODO: temporary fix fairseq dependency, remove after fairseq new release.
    from .fairseq_fp16_utils import Fairseq_FP16OptimizerMixin

# TODO: remove this try block after the new release by fairseq that
# contains the dependency
try:
    from fairseq.optim.fp16_optimizer import (
        _MemoryEfficientFP16OptimizerMixin as Fairseq_MemoryEfficientFP16OptimizerMixin,
    )
except ImportError:
    from .fairseq_fp16_utils import Fairseq_MemoryEfficientFP16OptimizerMixin

"""
Tips:
1. Recommand run fp16 on latest generation (Volta V100) GPU, CUDA 9.1 or newer
   to leverage tensor cores, which provide 8x more throughput than single
   precision math pipelines.
2. Additionally:
    - Batch size should be a multiple of 8
    - Tokens size should be a multiple of 8
    - Embedding layers should be padded to be a multiple of 8
    - Ideally, everything should be a multiple of 8 (e.g padding, etc)
3. Larger batch_size might increase GPU utilization and better performance.
"""


class FP16Optimizer(Optimizer):
    __EXPANSIBLE__ = True

    def __init__(self, fp32_optimizer):
        self.fp32_optimizer: torch.optim.Optimizer = fp32_optimizer

    @property
    def param_groups(self):
        return self.fp32_optimizer.param_groups

    def finalize(self) -> bool:
        return self.fp32_optimizer.finalize()

    # methods to implement
    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def step(self, closure=None):
        raise NotImplementedError

    def backward(self, loss):
        raise NotImplementedError

    def clip_grad_norm(self, max_norm, model):
        raise NotImplementedError

    def pre_export(self, model):
        raise NotImplementedError


"""
Apex amp: https://github.com/NVIDIA/apex/tree/master/apex/amp

FP32 Master Weights <--(step)-- FP32 Gradients <--(unscale)-- Scaled FP16 Gradients
       |                                                        |
(copy) |                                                        | (backprop)
       |                                                        |
FP16 Weights --(forward)--> FP32 Loss --(loss scaling)--> Scaled FP32 Loss

Using amp require adding three lines of code.
https://nvidia.github.io/apex/amp.html
1. Allow Amp to perform casts as required by the opt_level:
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

2. loss.backward() replace with:
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

3. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) replace with:
torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)

Opt level explaination (from Nvidia Apex):

* O1:  Insert automatic casts around Pytorch functions and Tensor methods
 - The type of your model's weights is not altered.  However, internally,
   Pytorch functions are patched to cast any Tensor Core-friendly ops to FP16
   for speed, while operations that might benefit from the additional stability
   of FP32 are patched to cast their inputs to fp32.
 - O1 is the safest way to try mixed precision training, and is recommended when
   trying mixed precision training for the first time.

* O2:  FP16 training with FP32 batchnorm and FP32 master weights.
 - Calls .half() on your model, converting the entire model (except for batchnorms)
   to FP16.  Batchnorms are retained in FP32 for additional stability.
 - The forward pass is patched to cast incoming Tensors to FP16, so you don't
   need to change your data pipeline.
 - O2 creates FP32 master weights outside the model and patches any optimizers
   to update these master weights, then copy the master weights into the FP16
   model weights.
"""


class FP16OptimizerApex(FP16Optimizer):
    class Config(FP16Optimizer.Config):
        # O1: Insert automatic casts around Pytorch functions and Tensor methods
        # O2: FP16 training with FP32 batchnorm and FP32 master weights. (recommand)
        opt_level: str = "O2"
        # initial loss scale, None will use the default loss_scale
        # defined in opt_level (for example: "dynamic" for O2)
        init_loss_scale: Optional[int] = None
        # determine the minimum loss scale
        min_loss_scale: Optional[float] = None

    def __init__(
        self,
        fp32_optimizer: Optimizer,
        model: torch.nn.Module,
        opt_level: str,
        init_loss_scale: Optional[int],
        min_loss_scale: Optional[float],
    ):
        assert precision.FP16_ENABLED and not _APEX_DISABLED
        model, fp32_optimizer = amp.initialize(
            model,
            fp32_optimizer,
            opt_level=opt_level,
            loss_scale=init_loss_scale,
            min_loss_scale=min_loss_scale,
        )

        super().__init__(fp32_optimizer)
        self.opt_level = opt_level

    @classmethod
    def from_config(
        cls,
        fp16_config: Config,
        model: torch.nn.Module,
        fp32_config: Optimizer.Config,
        *unused,
    ):
        fp32_optimizer = create_optimizer(fp32_config, model)
        return cls(
            fp32_optimizer,
            model,
            fp16_config.opt_level,
            fp16_config.init_loss_scale,
            fp16_config.min_loss_scale,
        )

    def state_dict(self):
        return self.fp32_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.fp32_optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.fp32_optimizer.zero_grad()

    def step(self, closure=None):
        self.fp32_optimizer.step(closure)

    def backward(self, loss):
        with amp.scale_loss(
            loss, self.fp32_optimizer, delay_unscale=precision.DELAY_UNSCALE
        ) as scaled_loss:
            scaled_loss.backward()

    def clip_grad_norm(self, max_norm, model):
        if max_norm is not None:
            return torch.nn.utils.clip_grad_norm_(
                amp.master_params(self.fp32_optimizer), max_norm
            )
        else:
            return None

    def pre_export(self, model):
        if self._opt_level == "O2":
            # convert model parameters back to fp32
            model.float()
        else:
            # restoring uncasted versions of functions
            amp._amp_state.handle._deactivate()

        precision.FP16_ENABLED = False


class MemoryEfficientFP16OptimizerFairseq(
    Fairseq_MemoryEfficientFP16OptimizerMixin, FP16Optimizer
):
    """
    Wrap the mem efficient *optimizer* to support FP16 (mixed precision) training.
    """

    class Config(FP16Optimizer.Config):
        # initial loss scale
        init_loss_scale: int = 2 ** 7
        # determine when to increase loss scale,
        # represents: consecutive number of non-overflow steps
        scale_window: Optional[int] = None
        # determine when to decrease loss scale, value range should be from 0 to 1,
        # represents: percentage of overflow since last rescale
        scale_tolerance: float = 0.0
        # determine the loss scale minimum value threshold
        threshold_loss_scale: Optional[float] = None
        # used to detect loss exploding, exception will be raised if loss_scale
        # reach this value
        min_loss_scale: float = 0.0001

    def __init__(
        self,
        fp16_params,
        optimizer,
        init_loss_scale,
        scale_window,
        scale_tolerance,
        threshold_loss_scale,
        min_loss_scale,
        num_accumulated_batches,
    ):
        assert precision.FP16_ENABLED
        super().__init__(optimizer)

        self.wrapped_optimizer = optimizer

        if scale_window is None:
            scale_window = (
                2 ** 14 / cuda.DISTRIBUTED_WORLD_SIZE / num_accumulated_batches
            )
        else:
            scale_window = scale_window

        self.scaler = Fairseq_DynamicLossScaler(
            init_scale=init_loss_scale,
            scale_window=scale_window,
            tolerance=scale_tolerance,
            threshold=threshold_loss_scale,
        )
        self.min_loss_scale = min_loss_scale

    @classmethod
    def from_config(
        cls,
        fp16_config: Config,
        model: torch.nn.Module,
        fp32_config: Optimizer.Config,
        num_accumulated_batches: int,
    ):
        model = model.half()
        fp16_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        fp32_optimizer = create_optimizer(fp32_config, model)
        print(
            "| Fairseq MemoryEfficientFP16Optimizer with init_loss_scale={}".format(
                fp16_config.init_loss_scale
            )
        )
        return cls(
            fp16_params=fp16_params,
            optimizer=fp32_optimizer,
            init_loss_scale=fp16_config.init_loss_scale,
            scale_window=fp16_config.scale_window,
            scale_tolerance=fp16_config.scale_tolerance,
            threshold_loss_scale=fp16_config.threshold_loss_scale,
            min_loss_scale=fp16_config.min_loss_scale,
            num_accumulated_batches=num_accumulated_batches,
        )

    def clip_grad_norm(self, max_norm, unused_model):
        # fairseq clip_grad_norm will skip clipping when max_norm is 0.
        if max_norm is None:
            max_norm = 0.0
        return super().clip_grad_norm(max_norm)

    def pre_export(self, model):
        model.float()
        precision.FP16_ENABLED = False


class FP16OptimizerFairseq(Fairseq_FP16OptimizerMixin, FP16Optimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.
    """

    class Config(FP16Optimizer.Config):
        # initial loss scale
        init_loss_scale: int = 2 ** 7
        # determine when to increase loss scale,
        # represents: consecutive number of non-overflow steps
        scale_window: Optional[int] = None
        # determine when to decrease loss scale, value range should be from 0 to 1,
        # represents: percentage of overflow since last rescale
        scale_tolerance: float = 0.0
        # determine the loss scale minimum value threshold
        threshold_loss_scale: Optional[float] = None
        # used to detect loss exploding, exception will be raised if loss_scale
        # reach this value
        min_loss_scale: float = 0.0001

    def __init__(
        self,
        fp16_params,
        fp32_optimizer,
        init_loss_scale,
        scale_window,
        scale_tolerance,
        threshold_loss_scale,
        min_loss_scale,
        num_accumulated_batches,
    ):
        assert precision.FP16_ENABLED
        super().__init__(fp32_optimizer)

        self.fp16_params = fp16_params
        self.fp32_params = self.build_fp32_params(fp16_params)

        if scale_window is None:
            scale_window = (
                2 ** 14 / cuda.DISTRIBUTED_WORLD_SIZE / num_accumulated_batches
            )
        else:
            scale_window = scale_window

        self.scaler = Fairseq_DynamicLossScaler(
            init_scale=init_loss_scale,
            scale_window=scale_window,
            tolerance=scale_tolerance,
            threshold=threshold_loss_scale,
        )
        self.min_loss_scale = min_loss_scale

        # reset fp32_optimizer param groups to using master weights
        fp32_param_group = self.fp32_optimizer.param_groups[0]
        fp32_param_group["params"] = [self.fp32_params]
        self.fp32_optimizer.param_groups = []
        self.fp32_optimizer.add_param_group(fp32_param_group)

    @classmethod
    def from_config(
        cls,
        fp16_config: Config,
        model: torch.nn.Module,
        fp32_config: Optimizer.Config,
        num_accumulated_batches: int,
    ):
        model = model.half()
        fp16_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        fp32_optimizer = create_optimizer(fp32_config, model)
        print(
            "| Fairseq FP16Optimizer with init_loss_scale={}".format(
                fp16_config.init_loss_scale
            )
        )
        return cls(
            fp16_params=fp16_params,
            fp32_optimizer=fp32_optimizer,
            init_loss_scale=fp16_config.init_loss_scale,
            scale_window=fp16_config.scale_window,
            scale_tolerance=fp16_config.scale_tolerance,
            threshold_loss_scale=fp16_config.threshold_loss_scale,
            min_loss_scale=fp16_config.min_loss_scale,
            num_accumulated_batches=num_accumulated_batches,
        )

    def clip_grad_norm(self, max_norm, unused_model):
        # fairseq clip_grad_norm will skip clipping when max_norm is 0.
        if max_norm is None:
            max_norm = 0.0
        return super().clip_grad_norm(max_norm)

    def pre_export(self, model):
        model.float()
        precision.FP16_ENABLED = False


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

Usage Example:
1    optim.zero_grad()
2    for i in range(N):
3        model.forward()  ---- fp16 weights
4        pre_process   ---- fp16 grads upscale
5        optim.backward() ---- upscaled fp16 grads
6        post_process   ---- downscale and float to fp32 grads
7    optim.step()         ---- fp32 weights and grads

class FP16_Optimizer:
Properties:
    inner_optimizer(torch.optim): optimizer in pytext (eg. Adam)
                    which is initialized with fp16 params already
    param_groups (list): list of dictionaries: key(string), value (list)
    loss_scaler(DynamicLossScaler): handle upscale, unscale, check_overflow
    weights_update_needed(bool): whether coping weights from master to model is needed
    grads_update_needed(bool): whether copying grads from model to master is needed

class DynamicLossScaler:
 properties:
    init_scale(int): beginning value of loss scale
    scale_factor(int): the step length that we use to increase the scale
    scale_window(int): the upper bound of iterations among which no overflow is triggered
    is_overflow(bool): indicate whether overflow happens in this step
    is_scaled(bool): whether grads are scaled
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

    def check_overflow_(self, grad):
        if grad is not None:
            cpu_sum = float(grad.float().sum())
            if (
                cpu_sum == float("inf")
                or cpu_sum == -float("inf")
                or cpu_sum != cpu_sum
            ):
                self.is_overflow = True
            else:
                self.is_overflow = False

    def check_overflow(self, params):
        self.is_overflow = False
        for p in generate_params(params):
            self.check_overflow_(p.grad)
            if self.is_overflow:
                break

    def update_scale(self):
        r"""According to overflow situation, adjust loss scale.

        Once overflow happened, we decrease the scale by scale_factor.
        Setting tolerance is another approach depending on cases.

        If we haven't had overflows for #scale_window times, we should increase
        the scale by scale_factor.
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


class FP16OptimizerDeprecated(object):
    def __init__(self, init_optimizer, init_scale, scale_factor, scale_window):
        r"""Initialize master weights maintaining optimizer.

        Args:
            init_optimizer(torch.optim.Optimizer): an initialized optimizer
            init_scale(int): beginning value of loss scale
            scale_factor(int): step that we adjust loss scale
            scale_window(int): tolerence for non-overflows

        Effects:
            Initialize the optimizer, create master weights copy and loss scaler.

        Modifies:
            Record the reference of model params (fp16).
            Change the inner optimizer's params to fp32.
            Initialized the scaler, state and default
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
        # print("-----running backward----")
        self.grads_update_needed = True
        return self.loss_scaler.upscale(loss)

    def step(self):
        r"""Realize weights update.

        Update the grads from model to master. During iteration for parameters,
        we check overflow after floating grads and copy. Then do unscaling.

        If overflow doesn't happen, call inner optimizer's step() and copy
        back the updated weights from inner optimizer to model.

        Update loss scale according to overflow checking result.
        """
        self._grads_from_model_to_master()
        if not self.loss_scaler.is_overflow:
            self.inner_optimizer.step()
            self.weights_update_needed = True
            self._weights_from_master_to_model()
        self.loss_scaler.update_scale()

    def _grads_from_model_to_master(self):
        r"""Sync grads from model to inner optimizer

        During each iteration, check overflow of grads.
        If not overflow, float the grads and copy to inner optimizer, unscale.
        """
        if self.grads_update_needed:
            for model_param, master_param in zip(
                generate_params(self.param_groups),
                generate_params(self.inner_optimizer.param_groups),
            ):
                # check master grad overflow
                self.loss_scaler.check_overflow_(model_param.grad)
                # print("checking overflow---{}".format(self.loss_scaler.is_overflow))
                if self.loss_scaler.is_overflow:
                    break

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
        self.inner_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.param_groups = state_dict["param_groups"]

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
        FP16OptimizerDeprecated(optimizer, init_scale, scale_factor, scale_window)
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


class PureFP16Optimizer(FP16OptimizerDeprecated):
    def __init__(
        self, init_optimizer, init_scale=2.0 ** 16, scale_factor=2, scale_window=2000
    ):
        r"""Initialize the memory-efficient optimizer

        Args:
            init_optimizer(torch.optim.Optimizer): an initialized optimizer
            init_scale(int): beginning value of loss scale
            scale_factor(int): step that we adjust loss scale
            scale_window(int): tolerence for non-overflows

        Effects:
            initialize this optimizer wrapper and loss scaling tools,
            initialized the scaler and state
        """
        self.inner_optimizer = init_optimizer
        self.param_groups = self.inner_optimizer.param_groups
        self.loss_scaler = DynamicLossScaler(init_scale, scale_factor, scale_window)
        self.state = self.inner_optimizer.state
        self.is_scaled = False
        print("===============Pure Memory Efficient Optimizer===============")

    def scale_loss(self, loss):
        r"""Scale the loss.

        Args:
            loss(pytext.Loss): loss function object
        """
        self.is_scaled = True
        return self.loss_scaler.upscale(loss)

    def step(self):
        r"""Updates the weights in inner optimizer.

        If inner optimizer supports memory efficient, check overflow,
        unscale and call advanced step.

        Otherwise, float weights and grads, check whether grads are overflow
        during the iteration, if not overflow, unscale grads and call inner
        optimizer's step; If overflow happens, do nothing, wait to the end
        to call half weights and grads (grads will be eliminated in zero_grad)
        """
        support = getattr(self.inner_optimizer, "supports_memory_efficient_fp16", False)
        if support:
            self.loss_scaler.check_overflow(self.param_groups)
            if not self.loss_scaler.is_overflow:
                self._unscale()
                self.inner_optimizer.step()
        else:
            self._fp16_to_fp32()
            if not self.loss_scaler.is_overflow:
                self.inner_optimizer.step()
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
                self.loss_scaler.check_overflow_(p.grad)
                if self.loss_scaler.is_overflow:
                    break
                self.loss_scaler.unscale(p.grad)

    def _fp32_to_fp16(self):
        for p in generate_params(self.param_groups):
            p.data = p.data.half()
            if p.grad is not None:
                p.grad.data = p.grad.data.half()

    def load_state_dict(self, state_dict):
        r"""Load an optimizer state dict.

        We prefer the configuration of the existing optimizer instance.
        Realize the same logic as in init() -- point the param_groups of outer
        optimizer to that of the inner_optimizer.
        """
        self.loss_scaler.scale = state_dict["loss_scale"]
        self.loss_scaler.is_overflow = state_dict["overflow"]
        self.inner_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.param_groups = self.inner_optimizer.param_groups
        self.state = self.inner_optimizer.state


class GeneratorFP16Optimizer(PureFP16Optimizer):
    def __init__(
        self, init_optimizer, init_scale=2.0 ** 16, scale_factor=2, scale_window=2000
    ):
        r"""Initialize the generator implementation method of memory efficient optimizer.

        Args:
            init_optimizer(torch.optim.Optimizer): an initialized optimizer
            init_scale(int): beginning value of loss scale
            scale_factor(int): step that we adjust loss scale
            scale_window(int): tolerence for non-overflows

        Effects:
            We create another copy of references of parameters in self.param_groups
            to keep trace of changed weights and grads.
        """
        self.inner_optimizer = init_optimizer
        self.param_groups = []
        for group in self.inner_optimizer.param_groups:
            fp16_group = {}
            for key, value in group.items():
                fp16_group[key] = value
            self.param_groups.append(fp16_group)

        self.loss_scaler = DynamicLossScaler(init_scale, scale_factor, scale_window)
        self.state = self.inner_optimizer.state
        self.is_scaled = False
        print("=============Generator Memory Efficient Optimizer==============")

    def step(self):
        r"""Updates weights.

        Effects:
            Check overflow, if not, when inner_optimizer supports memory-effcient
            step, do overall unscale and call memory-efficient step.

            If it doesn't support, modify each parameter list in param_groups
            of inner_optimizer to a generator of the tensors. Call normal step
            then, data type changing will be added automatically in that function.

            No matter whether it is overflow, we need to update scale at the
            last step.
        """
        support = getattr(self.inner_optimizer, "supports_memory_efficient_fp16", False)

        self.loss_scaler.check_overflow(self.param_groups)
        if not self.loss_scaler.is_overflow:
            if support:
                self._unscale()
                self.inner_optimizer.step()
            else:
                self._preprocess_step()
                self.inner_optimizer.step()

        self.loss_scaler.update_scale()

    def _preprocess_step(self):
        r"""Change the parameter list to a generator.
        """
        for i, group in enumerate(self.param_groups):
            self.inner_optimizer.param_groups[i]["params"] = convert_generator(
                group["params"], self.loss_scaler.scale
            )

    def load_state_dict(self, state_dict):
        r"""Load an optimizer state dict.

        We prefer the configuration of the existing optimizer instance.
        After we load state dict to inner_optimizer, we create the copy of
        references of parameters again as in init().
        """
        self.loss_scaler.scale = state_dict["loss_scale"]
        self.loss_scaler.is_overflow = state_dict["overflow"]
        self.inner_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.param_groups = []
        for group in self.inner_optimizer.param_groups:
            fp16_group = {}
            for key, value in group.items():
                fp16_group[key] = value
            self.param_groups.append(fp16_group)
        self.state = self.inner_optimizer.state


def convert_generator(params, scale):
    r"""Create the generator for parameter tensors.

    For each parameter, we float and unscale it. After the caller calls next(),
    we realize the half process and start next parameter's processing.
    """
    for p in params:
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
            p.grad.div_(scale)
        yield p
        p.data = p.data.half()
        if p.grad is not None:
            p.grad.data = p.grad.data.half()
