#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# DIRECTLY COPIED OVER FROM pytorch/contrib
# https://raw.githubusercontent.com/pytorch/contrib/master/test/test_swa.py

import contextlib
import functools
import re
import warnings
from unittest import TestCase

import torch
from pytext.optimizer import Lamb, StochasticWeightAveraging
from torch import nn, optim, sparse
from torch.autograd import Variable
from torch.utils import data


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.DoubleTensor(
        (-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2))
    )


def wrap_old_fn(old_fn, **config):
    def wrapper(closure, params, state):
        return old_fn(closure, params, config, state)

    return wrapper


def equal(x, y, prec=1e-4):
    return torch.all(torch.lt(torch.abs(torch.add(x, -y)), prec))


class TestSWA(TestCase):
    # Pavel: I slightly update the _test_... functions to (1) remove the
    # legacy-related parts and (2) add oprimizer.finalize() in the end of
    # optimization

    @contextlib.contextmanager
    def assertWarnsRegex(self, regex, msg=""):
        r"""
        As a context manager, test if wrapped code raises any warning with
         message that contains the regex pattern :attr:`regex`.
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            yield
            self.assertTrue(len(ws) > 0, msg)
            found = any(re.search(regex, str(w.message)) is not None for w in ws)
            self.assertTrue(found, msg)

    def _test_rosenbrock(self, constructor, automode=True):
        # automode shows wether we need to update SWA params manually

        params = torch.tensor([1.5, 1.5], requires_grad=True)
        optimizer = constructor([params])

        solution = torch.tensor([1.0, 1.0])
        initial_dist = params.data.dist(solution)

        def eval():
            # SWA
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            # loss.backward() will give **slightly** different
            # gradients, than drosenbtock, because of a different ordering
            # of floating point operations. In most cases it doesn't matter,
            # but some optimizers are so sensitive that they can temporarily
            # diverge up to 1e-4, just to converge again. This makes the
            # comparison more stable.
            params.grad.data.copy_(drosenbrock(params.data))
            return loss

        for _ in range(2000):
            optimizer.step(eval)
            if not automode:
                optimizer.update_swa()
        optimizer.finalize()

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_rosenbrock_sparse(self, constructor, sparse_only=False):
        params = torch.tensor([1.5, 1.5], requires_grad=True)
        optimizer = constructor([params])

        if not sparse_only:
            params_c = params.detach().clone().requires_grad_()
            optimizer_c = constructor([params_c])

        solution = torch.tensor([1.0, 1.0])
        initial_dist = params.data.dist(solution)

        def eval(params, sparse_grad, w):
            # Depending on w, provide only the x or y gradient
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            grad = drosenbrock(params.data)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor
            if w:
                i = torch.LongTensor([[0, 0]])
                x = grad[0]
                v = torch.DoubleTensor([x / 4.0, x - x / 4.0])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.DoubleTensor([y - y / 4.0, y / 4.0])
            x = sparse.DoubleTensor(i, v, torch.Size([2]))
            if sparse_grad:
                params.grad.data = x
            else:
                params.grad.data = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                assert equal(params.data, params_c.data)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_basic_cases_template(self, weight, bias, input, constructor):
        weight = weight.requires_grad_()
        bias = bias.requires_grad_()
        optimizer = constructor(weight, bias)

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().item()
        for _ in range(200):
            optimizer.step(fn)
        self.assertLess(fn().item(), initial_value)

    def _test_basic_cases(self, constructor, ignore_multidevice=False):
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor,
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor,
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor,
        )

    def _build_params_dict(self, weight, bias, **kwargs):
        return [dict(params=[weight]), dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    # Test SWA
    # keep timing out in diffs T54574493
    def DISABLED_test_swa(self):
        def sgd_constructor(params):
            sgd = optim.SGD(params, lr=1e-3)
            return StochasticWeightAveraging(
                sgd, swa_start=1000, swa_freq=1, swa_lr=1e-3
            )

        def sgd_manual_constructor(params):
            sgd = optim.SGD(params, lr=1e-3)
            return StochasticWeightAveraging(sgd)

        def sgd_momentum_constructor(params):
            sgd = optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)
            return StochasticWeightAveraging(
                sgd, swa_start=1000, swa_freq=1, swa_lr=1e-3
            )

        def adam_constructor(params):
            adam = optim.Adam(params, lr=1e-2)
            return StochasticWeightAveraging(
                adam, swa_start=1000, swa_freq=1, swa_lr=1e-2
            )

        def adadelta_constructor(params):
            adadelta = optim.Adadelta(params)
            return StochasticWeightAveraging(adadelta, swa_start=1000, swa_freq=1)

        def adagrad_constructor(params):
            adagrad = optim.Adagrad(params, lr=1e-1)
            return StochasticWeightAveraging(
                adagrad, swa_start=1000, swa_freq=1, swa_lr=1e-2
            )

        def adamax_constructor(params):
            adamax = optim.Adamax(params, lr=1e-1)
            return StochasticWeightAveraging(
                adamax, swa_start=1000, swa_freq=1, swa_lr=1e-2
            )

        def rmsprop_constructor(params):
            rmsprop = optim.RMSprop(params, lr=1e-2)
            return StochasticWeightAveraging(
                rmsprop, swa_start=1000, swa_freq=1, swa_lr=1e-3
            )

        def rprop_constructor(params):
            rprop = optim.Rprop(params, lr=1e-2)
            return StochasticWeightAveraging(
                rprop, swa_start=1000, swa_freq=1, swa_lr=1e-3
            )

        def asgd_constructor(params):
            asgd = optim.ASGD(params, lr=1e-3)
            return StochasticWeightAveraging(
                asgd, swa_start=1000, swa_freq=1, swa_lr=1e-3
            )

        def lbfgs_constructor(params):
            lbfgs = optim.LBFGS(params, lr=5e-2, max_iter=5)
            return StochasticWeightAveraging(
                lbfgs, swa_start=1000, swa_freq=1, swa_lr=1e-3
            )

        auto_constructor_list = [
            sgd_constructor,
            sgd_momentum_constructor,
            adam_constructor,
            adadelta_constructor,
            adagrad_constructor,
            adamax_constructor,
            rmsprop_constructor,
            rprop_constructor,
            asgd_constructor,
            lbfgs_constructor,
        ]

        for i, constructor in enumerate(auto_constructor_list):
            self._test_rosenbrock(constructor)
            self._test_basic_cases(lambda weight, bias: constructor([weight, bias]))
            if i < len(auto_constructor_list) - 1:
                self._test_basic_cases(
                    lambda weight, bias: constructor(
                        self._build_params_dict(weight, bias, lr=1e-2)
                    )
                )
                self._test_basic_cases(
                    lambda weight, bias: constructor(
                        self._build_params_dict_single(weight, bias, lr=1e-2)
                    )
                )

        self._test_rosenbrock(sgd_manual_constructor, automode=False)

    def _define_vars_loss_opt(self):
        x = Variable(torch.Tensor([5.0, 2.0]), requires_grad=True)
        y = Variable(torch.Tensor([3.0, 7.0]), requires_grad=True)

        def loss_fun(a, b):
            return torch.sum(a * b) ** 2

        opt = optim.SGD(
            [{"params": [x]}, {"params": [y], "lr": 1e-3}], lr=1e-2, momentum=0.9
        )
        return x, y, loss_fun, opt

    @staticmethod
    def _update_test_vars(i, swa_freq, swa_start, n_avg, x_sum, y_sum, x, y, upd_fun):
        if i % swa_freq == 0 and i > swa_start:
            upd_fun()
            n_avg += 1
            x_sum += x.data
            y_sum += y.data
        return n_avg, x_sum, y_sum

    def test_swa_auto(self):
        # Tests SWA in Auto mode: values of x and y after opt.finalize()
        # should be equal to the manually computed averages
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        swa_start = 5
        swa_freq = 2
        opt = StochasticWeightAveraging(
            opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=0.001
        )

        x_sum = torch.zeros_like(x)
        y_sum = torch.zeros_like(y)
        n_avg = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            n_avg, x_sum, y_sum = self._update_test_vars(
                i, swa_freq, swa_start, n_avg, x_sum, y_sum, x, y, upd_fun=lambda: None
            )

        opt.finalize()
        x_avg = x_sum / n_avg
        y_avg = y_sum / n_avg
        assert equal(x_avg, x)
        assert equal(y_avg, y)

    def test_swa_manual(self):
        # Tests SWA in manual mode: values of x and y after opt.finalize()
        # should be equal to the manually computed averages
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        opt = StochasticWeightAveraging(opt)
        swa_start = 5
        swa_freq = 2

        x_sum = torch.zeros_like(x)
        y_sum = torch.zeros_like(y)
        n_avg = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            n_avg, x_sum, y_sum = self._update_test_vars(
                i,
                swa_freq,
                swa_start,
                n_avg,
                x_sum,
                y_sum,
                x,
                y,
                upd_fun=opt.update_swa,
            )

        opt.finalize()
        x_avg = x_sum / n_avg
        y_avg = y_sum / n_avg
        assert equal(x_avg, x)
        assert equal(y_avg, y)

    def test_swa_manual_group(self):
        # Tests SWA in manual mode with only y param group updated:
        # value of x should not change after opt.finalize() and y should
        # be equal to the manually computed average
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        opt = StochasticWeightAveraging(opt)
        swa_start = 5
        swa_freq = 2

        y_sum = torch.zeros_like(y)
        n_avg = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            n_avg, _, y_sum = self._update_test_vars(
                i,
                swa_freq,
                swa_start,
                n_avg,
                0,
                y_sum,
                x,
                y,
                upd_fun=lambda: opt.update_swa_group(opt.param_groups[1]),
            )

        x_before_swap = x.data.clone()

        with self.assertWarnsRegex(
            re.escape(r"SWA wasn't applied to param {}".format(x))
        ):
            opt.finalize()

        y_avg = y_sum / n_avg
        assert equal(y_avg, y)
        assert equal(x_before_swap, x)

    def test_swa_auto_group_added_during_run(self):
        # Tests SWA in Auto mode with the second param group added after several
        # optimizations steps. The expected behavior is that the averaging for
        # the second param group starts at swa_start steps after it is added.
        # For the first group averaging should start swa_start steps after the
        # first step of the optimizer.

        x, y, loss_fun, _ = self._define_vars_loss_opt()
        opt = optim.SGD([x], lr=1e-3, momentum=0.9)
        swa_start = 5
        swa_freq = 2
        opt = StochasticWeightAveraging(
            opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=0.001
        )

        x_sum = torch.zeros_like(x)
        y_sum = torch.zeros_like(y)
        x_n_avg = 0
        y_n_avg = 0
        x_step = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            x_step += 1
            if i % swa_freq == 0 and i > swa_start:
                x_n_avg += 1
                x_sum += x.data

        x_avg = x_sum / x_n_avg

        opt.add_param_group({"params": y, "lr": 1e-4})

        for y_step in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            x_step += 1
            if y_step % swa_freq == 0 and y_step > swa_start:
                y_n_avg += 1
                y_sum += y.data
            if x_step % swa_freq == 0 and x_step > swa_start:
                x_n_avg += 1
                x_sum += x.data
                x_avg = x_sum / x_n_avg

        opt.finalize()
        x_avg = x_sum / x_n_avg
        y_avg = y_sum / y_n_avg
        assert equal(x_avg, x)
        assert equal(y_avg, y)

    def test_swa_lr(self):
        # Tests SWA learning rate: in auto mode after swa_start steps the
        # learning rate should be changed to swa_lr; in manual mode swa_lr
        # must be ignored

        # Auto mode
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        swa_start = 5
        swa_freq = 2
        initial_lr = opt.param_groups[0]["lr"]
        swa_lr = initial_lr * 0.1
        opt = StochasticWeightAveraging(
            opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr
        )

        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            lr = opt.param_groups[0]["lr"]
            if i > swa_start:
                assert equal(lr, swa_lr)
            else:
                assert equal(lr, initial_lr)

        # Manual Mode
        x, y, loss, opt = self._define_vars_loss_opt()
        initial_lr = opt.param_groups[0]["lr"]
        swa_lr = initial_lr * 0.1
        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = StochasticWeightAveraging(opt, swa_lr=swa_lr)

        for _ in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            lr = opt.param_groups[0]["lr"]
            assert equal(lr, initial_lr)

    def test_swa_auto_mode_detection(self):
        # Tests that SWA mode (auto or manual) is chosen correctly based on
        # parameters provided

        # Auto mode
        x, y, loss_fun, base_opt = self._define_vars_loss_opt()
        swa_start = 5
        swa_freq = 2
        swa_lr = 0.001

        opt = StochasticWeightAveraging(
            base_opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr
        )
        assert equal(opt._auto_mode, True)

        opt = StochasticWeightAveraging(
            base_opt, swa_start=swa_start, swa_freq=swa_freq
        )
        assert equal(opt._auto_mode, True)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = StochasticWeightAveraging(
                base_opt, swa_start=swa_start, swa_lr=swa_lr
            )
            assert equal(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = StochasticWeightAveraging(base_opt, swa_freq=swa_freq, swa_lr=swa_lr)
            assert equal(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = StochasticWeightAveraging(base_opt, swa_start=swa_start)
            assert equal(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = StochasticWeightAveraging(base_opt, swa_freq=swa_freq)
            assert equal(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = StochasticWeightAveraging(base_opt, swa_lr=swa_lr)
            assert equal(opt._auto_mode, False)

    def test_swa_raises(self):
        # Tests that SWA raises errors for wrong parameter values

        x, y, loss_fun, opt = self._define_vars_loss_opt()

        with self.assertRaisesRegex(ValueError, "Invalid SWA learning rate: -0.0001"):
            opt = StochasticWeightAveraging(opt, swa_start=1, swa_freq=2, swa_lr=-1e-4)

        with self.assertRaisesRegex(ValueError, "Invalid swa_freq: 0"):
            opt = StochasticWeightAveraging(opt, swa_start=1, swa_freq=0, swa_lr=1e-4)

        with self.assertRaisesRegex(ValueError, "Invalid swa_start: -1"):
            opt = StochasticWeightAveraging(opt, swa_start=-1, swa_freq=0, swa_lr=1e-4)

    # bn_update test

    def _test_bn_update(self, data_tensor, dnn, device, label_tensor=None):
        class DatasetFromTensors(data.Dataset):
            def __init__(self, X, y=None):
                self.X = X
                self.y = y
                self.N = self.X.shape[0]

            def __getitem__(self, index):
                x = self.X[index]
                if self.y is None:
                    return x
                else:
                    y = self.y[index]
                    return x, y

            def __len__(self):
                return self.N

        with_y = label_tensor is not None
        ds = DatasetFromTensors(data_tensor, y=label_tensor)
        dl = data.DataLoader(ds, batch_size=5, shuffle=True)

        preactivation_sum = torch.zeros(dnn.n_features, device=device)
        preactivation_squared_sum = torch.zeros(dnn.n_features, device=device)
        total_num = 0
        for x in dl:
            if with_y:
                x, _ = x
            x = x.to(device)

            dnn(x)
            preactivations = dnn.compute_preactivation(x)
            if len(preactivations.shape) == 4:
                preactivations = preactivations.transpose(1, 3)
            preactivations = preactivations.reshape(-1, dnn.n_features)
            total_num += preactivations.shape[0]

            preactivation_sum += torch.sum(preactivations, dim=0)
            preactivation_squared_sum += torch.sum(preactivations ** 2, dim=0)

        preactivation_mean = preactivation_sum / total_num
        preactivation_var = preactivation_squared_sum / total_num
        preactivation_var = preactivation_var - preactivation_mean ** 2

        swa = StochasticWeightAveraging(optim.SGD(dnn.parameters(), lr=1e-3))
        swa.bn_update(dl, dnn, device=device)
        assert equal(preactivation_mean, dnn.bn.running_mean)
        assert equal(preactivation_var, dnn.bn.running_var, prec=1e-1)

    def test_bn_update(self):
        def test(net_cls, x_shape, y_shape, device):
            x = torch.rand(x_shape, device=device)
            y = torch.rand(y_shape, device=device)

            dnn = net_cls().to(device)
            orig_momentum = dnn.bn.momentum
            dnn.train()
            self._test_bn_update(x, dnn, device)
            self._test_bn_update(x, dnn, device, label_tensor=y)
            self.assertTrue(dnn.training)

            # check that bn_update preserves eval mode
            dnn.eval()
            self._test_bn_update(x, dnn, device)
            self.assertFalse(dnn.training)

            # check that momentum is preserved
            assert equal(dnn.bn.momentum, orig_momentum)

        # Test bn_update for fully-connected and convolutional networks with
        # BatchNorm1d and BatchNorm2d respectively
        objects = 100
        input_features = 5

        class DNN(nn.Module):
            def __init__(self):
                super(DNN, self).__init__()
                self.n_features = 100
                self.fc1 = nn.Linear(input_features, self.n_features)
                self.bn = nn.BatchNorm1d(self.n_features)

            def compute_preactivation(self, x):
                return self.fc1(x)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn(x)
                return x

        test(DNN, (objects, input_features), objects, "cpu")
        if torch.cuda.is_available():
            test(DNN, (objects, input_features), objects, "cuda")

        # Test bn_update for convolutional network and BatchNorm2d
        objects = 100
        channels = 3
        height, width = 5, 5

        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.n_features = 10
                self.conv1 = nn.Conv2d(
                    channels, self.n_features, kernel_size=3, padding=1
                )
                self.bn = nn.BatchNorm2d(self.n_features, momentum=0.3)

            def compute_preactivation(self, x):
                return self.conv1(x)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn(x)
                return x

        test(CNN, (objects, channels, height, width), objects, "cpu")
        if torch.cuda.is_available():
            test(CNN, (objects, channels, height, width), objects, "cuda")

    def test_lamb(self):
        def lamb_constructor(params):
            return StochasticWeightAveraging(
                Lamb(params, weight_decay=0.01), swa_start=1000, swa_freq=1, swa_lr=1e-2
            )

        self._test_rosenbrock(lamb_constructor, automode=False)
