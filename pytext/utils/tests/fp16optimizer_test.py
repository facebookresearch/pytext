#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import unittest

import torch
from pytext.optimizer import fp16_optimizer
from torch.autograd import Variable


""" unit tests for fp16Optimizer
- test_init: check data type of master and model params after initialization
- test_zero_grad: after initializing grad of model params, call zero_grad and
        check data type and value
- test_grads_from_model_to_master/test_weights_from_master_to_model: change grads
        in model/weights in master to make those two parts different, then call
        the sync function and check data type and equality of relative grads/weights
- test_combo: stepped stages of two entire consecutive iterations (cycle zero_grads,
        backward, step), after each being-tested function call, check data type
        and expected value
- test_flow_loss_decrease: imitate the whole procedure of calling a fp16optimizer
        in training workflow, providing weights, dataset, loss. Compare that with
        two normal optimizers in which one takes cares of weights and another take
        in charge of the grads. Check whether those two cases have the same values.
"""


class TestFp16Optimizer(unittest.TestCase):
    def _generate_inner_optim(self):
        param_groups = []
        tensors = []
        rand_in = 4
        for _iter in range(rand_in):
            a = torch.randn(rand_in, rand_in, requires_grad=True).half()
            b = torch.nn.Parameter(a)
            tensors.append(b)
        group = {"params": tensors}
        param_groups.append(group)
        self.optim = torch.optim.Adam(param_groups, lr=0.01, weight_decay=0.00001)
        self.fake_model = torch.randn(rand_in, rand_in, requires_grad=True)
        self.fake_model, self.fp16_optim = fp16_optimizer.initialize(
            self.fake_model, self.optim, "fake_opt", init_scale=2 ** 6
        )

    def test_init(self):
        self._generate_inner_optim()
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)

    def test_zero_grad(self):
        self._generate_inner_optim()
        self._pseudo_backward(self.fp16_optim.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self.fp16_optim.zero_grad()
        self._is_all_zero(self.fp16_optim.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)

    def test_grads_from_model_to_master(self):
        self._generate_inner_optim()
        self._pseudo_backward(self.fp16_optim.param_groups)
        # check data type before calling sync grads
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self.fp16_optim.grads_update_needed = True
        self.fp16_optim._grads_from_model_to_master()
        self._is_all_half(self.fp16_optim.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)

        for p in self._generate_params(self.fp16_optim.inner_optimizer.param_groups):
            p.grad.data.mul_(self.fp16_optim.loss_scaler.scale)

        self._is_grad_equal(
            self.fp16_optim.param_groups, self.fp16_optim.inner_optimizer.param_groups
        )
        self._is_all_half(self.fp16_optim.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)

    def test_weights_from_master_to_model(self):
        self._generate_inner_optim()
        self._pseudo_step(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self.fp16_optim.weights_update_needed = True
        self.fp16_optim._weights_from_master_to_model()
        self._is_all_half(self.fp16_optim.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_weight_equal(
            self.fp16_optim.param_groups, self.fp16_optim.inner_optimizer.param_groups
        )

    def test_two_iterations(self):
        self._generate_inner_optim()
        self._pseudo_backward(self.fp16_optim.param_groups)
        self.fp16_optim.grads_update_needed = True
        self.fp16_optim._grads_from_model_to_master()
        for p in self._generate_params(self.fp16_optim.inner_optimizer.param_groups):
            p.grad.data.mul_(self.fp16_optim.loss_scaler.scale)
        self._pseudo_step(self.fp16_optim.inner_optimizer.param_groups)
        self.fp16_optim.weights_update_needed = True
        self.fp16_optim._weights_from_master_to_model()
        self.fp16_optim.zero_grad()
        self._pseudo_backward(self.fp16_optim.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self.fp16_optim.grads_update_needed = True
        self.fp16_optim._grads_from_model_to_master()
        for p in self._generate_params(self.fp16_optim.inner_optimizer.param_groups):
            p.grad.data.mul_(self.fp16_optim.loss_scaler.scale)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self._is_grad_equal(
            self.fp16_optim.param_groups, self.fp16_optim.inner_optimizer.param_groups
        )
        self._pseudo_step_pair(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self.fp16_optim.weights_update_needed = True
        self.fp16_optim._weights_from_master_to_model()
        self._is_all_float(self.fp16_optim.inner_optimizer.param_groups)
        self._is_all_half(self.fp16_optim.param_groups)
        self._is_weight_equal(
            self.fp16_optim.param_groups, self.fp16_optim.inner_optimizer.param_groups
        )

    def _test_logic_prepare(self):
        N, D_in, H, D_out = 64, 1000, 100, 10
        x1 = Variable(torch.randn(N, D_in), requires_grad=False)
        x2 = Variable(torch.randn(N, D_in), requires_grad=False)
        x3 = Variable(torch.randn(N, D_in), requires_grad=False)
        y1 = Variable(torch.randn(N, D_out), requires_grad=False)
        y2 = Variable(torch.randn(N, D_out), requires_grad=False)
        y3 = Variable(torch.randn(N, D_out), requires_grad=False)
        self.data_set = [[x1, y1], [x2, y2], [x3, y3]]

        w1 = Variable(torch.randn(D_in, H), requires_grad=True)
        w1_data = copy.deepcopy(w1.data)
        w1_copy = Variable(w1_data, requires_grad=True)
        ww1_data = copy.deepcopy(w1.data)
        ww1 = Variable(ww1_data, requires_grad=True)
        fake_w1_data = copy.deepcopy(w1.data)

        w2 = Variable(torch.randn(H, D_out), requires_grad=True)
        w2_data = copy.deepcopy(w2.data)
        w2_copy = Variable(w2_data, requires_grad=True)
        ww2_data = copy.deepcopy(w2.data)
        ww2 = Variable(ww2_data, requires_grad=True)
        fake_w2_data = copy.deepcopy(w2.data)

        self.parameters = [w1, w2]
        parameters_copy = [w1_copy, w2_copy]
        learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        # for updating grads
        self.inner_optimizer = torch.optim.Adam(parameters_copy, lr=learning_rate)
        # for updating weights
        rand_in = 4
        fake_parameters = [fake_w1_data, fake_w2_data]
        fake_optimizer = torch.optim.Adam(fake_parameters, lr=learning_rate)
        self.fake_model = torch.randn(rand_in, rand_in, requires_grad=True)
        self.fake_model, self.fake_fp16_optim = fp16_optimizer.initialize(
            self.fake_model, fake_optimizer, "fake_opt", init_scale=2 ** 10
        )

        self.pparameters = [ww1, ww2]
        self.ooptimizer = torch.optim.Adam(self.pparameters, lr=learning_rate)
        self.fake_model = torch.randn(rand_in, rand_in, requires_grad=True)
        self.fake_model, self.ffp16_optim = fp16_optimizer.initialize(
            self.fake_model, self.ooptimizer, "fake_opt", init_scale=2 ** 10
        )

    def _test_logic_baseline(self):
        for _t in range(500):
            self.optimizer.zero_grad()
            for pair in self.data_set:
                y_pred = (
                    pair[0].mm(self.parameters[0]).clamp(min=0).mm(self.parameters[1])
                )
                loss = (y_pred - pair[1]).pow(
                    2
                ).sum() * self.fake_fp16_optim.loss_scaler.scale
                loss = loss / len(self.data_set)
                loss.backward()

            # from w1, w2 to w1_copy, w2_copy
            for model, master in zip(
                self._generate_params(self.optimizer.param_groups),
                self._generate_params(self.inner_optimizer.param_groups),
            ):
                if master.grad is None:
                    master.grad = torch.empty_like(master)
                master.grad.copy_(model.grad)

            # step
            self.fake_fp16_optim.loss_scaler.check_overflow(self.optimizer.param_groups)
            if not self.fake_fp16_optim.loss_scaler.is_overflow:
                for p in self._generate_params(self.inner_optimizer.param_groups):
                    p.grad.data.div_(self.fake_fp16_optim.loss_scaler.scale)
                self.inner_optimizer.step()

                # copy weight back
                for model, master in zip(
                    self._generate_params(self.optimizer.param_groups),
                    self._generate_params(self.inner_optimizer.param_groups),
                ):
                    model.data.copy_(master.data)
                self.fake_fp16_optim.loss_scaler.update_scale()

    def _test_logic_experiment(self):
        for _t in range(500):
            self.ffp16_optim.zero_grad()
            for pair in self.data_set:
                yy_pred = (
                    pair[0].mm(self.pparameters[0]).clamp(min=0).mm(self.pparameters[1])
                )
                lloss = (yy_pred - pair[1]).pow(2).sum()
                lloss = lloss / len(self.data_set)
                with fp16_optimizer.scale_loss(
                    lloss, self.ffp16_optim, delay_unscale=False
                ) as scaled_loss:
                    scaled_loss.backward()
            # Run backprop
            # Check for overflow
            self.ffp16_optim.step()

    def test_logic(self):
        self._test_logic_prepare()
        cpu_sum_base = []
        for p in self._generate_params(self.optimizer.param_groups):
            cpu_sum_base_init = float(p.data.float().sum())
            cpu_sum_base.append(cpu_sum_base_init)
        self._test_logic_baseline()
        for p in self._generate_params(self.optimizer.param_groups):
            cpu_sum_base_final = float(p.data.float().sum())
            cpu_sum_base.append(cpu_sum_base_final)

        cpu_sum_exp = []
        for p in self._generate_params(self.ffp16_optim.param_groups):
            cpu_sum_exp_init = float(p.data.float().sum())
            cpu_sum_exp.append(cpu_sum_exp_init)
        self._test_logic_experiment()
        for p in self._generate_params(self.ffp16_optim.param_groups):
            cpu_sum_exp_final = float(p.data.float().sum())
            cpu_sum_exp.append(cpu_sum_exp_final)

        print(cpu_sum_base, cpu_sum_exp)
        for i, j in zip(cpu_sum_base, cpu_sum_exp):
            self.assertEqual(i, j)

    def _is_all_float(self, master_param_groups):
        for p in self._generate_params(master_param_groups):
            self.assertEqual(p.dtype, torch.float32)

    def _is_all_half(self, model_param_groups):
        for p in self._generate_params(model_param_groups):
            self.assertEqual(p.dtype, torch.float16)

    def _is_grad_equal(self, model_param_groups, master_param_groups):
        for u, v in zip(
            self._generate_params(model_param_groups),
            self._generate_params(master_param_groups),
        ):
            for m, n in zip(u.grad.data, v.grad.data):
                for x, y in zip(m, n):
                    self.assertEqual(x.float(), y.float())

    def _is_weight_equal(self, model_param_groups, master_param_groups):
        for u, v in zip(
            self._generate_params(model_param_groups),
            self._generate_params(master_param_groups),
        ):
            for m, n in zip(u.data, v.data):
                for x, y in zip(m, n):
                    self.assertEqual(x.float(), y.float())

    def _is_all_zero(self, param_groups):
        for p in self._generate_params(param_groups):
            for u in p.grad.data:
                for v in u:
                    self.assertEqual(v.float(), 0.0)

    def _pseudo_backward(self, param_groups):
        for p in self._generate_params(param_groups):
            p.grad = None
            p = p.float()
            y = p * p * 3
            z = y.mean()
            z.backward()

    def _pseudo_step(self, param_groups):
        for p in self._generate_params(param_groups):
            p.data[1][1] = 4

    def _pseudo_step_pair(self, param_groups):
        for p in self._generate_params(param_groups):
            p.data[1][1] = 3

    def _generate_params(self, param_groups):
        for group in param_groups:
            for p in group["params"]:
                yield p
