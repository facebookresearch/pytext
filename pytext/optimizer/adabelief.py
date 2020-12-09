#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
from torch.optim.optimizer import Optimizer as PT_Optimizer

from .optimizers import Optimizer


class AdaBelief(Optimizer, PT_Optimizer):
    """
    `AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients`
    Paper: https://arxiv.org/abs/2010.07468
    Implementation has been copied over from the original author (https://github.com/juntang-zhuang/Adabelief-Optimizer)
    """

    class Config(Optimizer.Config):
        lr: float = 1e-3
        beta_1: float = 0.9
        beta_2: float = 0.999
        eps: float = 1e-8
        weight_decay: float = 0
        amsgrad: bool = False
        weight_decouple: bool = True
        fixed_decay: bool = True
        rectify: bool = False

    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module):
        return cls(
            params=model.parameters(),
            lr=config.lr,
            betas=(config.beta_1, config.beta_2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
            weight_decouple=config.weight_decouple,
            fixed_decay=config.fixed_decay,
            rectify=config.rectify,
        )

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        weight_decouple=False,
        fixed_decay=False,
        rectify=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        PT_Optimizer.__init__(self, params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print("Weight decoupling enabled in AdaBelief")
            if self.fixed_decay:
                print("Weight decay fixed")
        if self.rectify:
            print("Rectification enabled in AdaBelief")
        if amsgrad:
            print("AMS enabled in AdaBelief")

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                amsgrad = group["amsgrad"]

                # State initialization
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format
                )

                # Exponential moving average of squared gradient values
                state["exp_avg_var"] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_var"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

    def step(self, closure=None, **kwargs):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaBelief does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["rho_inf"] = 2.0 / (1.0 - beta2) - 1.0
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_var"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_var"] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )

                # get current state variable
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        p.data.mul_(1.0 - group["weight_decay"])
                else:
                    if group["weight_decay"] != 0:
                        grad.add_(group["weight_decay"], p.data)

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    1 - beta2, grad_residual, grad_residual
                )

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (
                        max_exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])
                else:
                    denom = (
                        exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])

                if not self.rectify:
                    # Default update
                    step_size = group["lr"] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                else:  # Rectified update
                    # calculate rho_t
                    state["rho_t"] = state["rho_inf"] - 2 * state[
                        "step"
                    ] * beta2 ** state["step"] / (1.0 - beta2 ** state["step"])

                    if (
                        state["rho_t"] > 4
                    ):  # perform Adam style update if variance is small
                        rho_inf, rho_t = state["rho_inf"], state["rho_t"]
                        rt = (
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / (rho_inf - 4.0)
                            / (rho_inf - 2.0)
                            / rho_t
                        )
                        rt = math.sqrt(rt)

                        step_size = rt * group["lr"] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:  # perform SGD style update
                        p.data.add_(-group["lr"], exp_avg)

        return loss

    def clip_grad_norm(self, max_norm, model=None):
        return Optimizer.clip_grad_norm(self, max_norm, model)
