#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# This module is a copy of DistributedDataParallel before it was
# switched to use the new reducer code in #18953.
# We are debugging a problem where the loss becomes NaN after this
# change was landed and need to narrow it down.

import copy

import torch
import torch.distributed as dist
from torch.cuda._utils import _get_device_index
from torch.cuda.comm import broadcast_coalesced
from torch.nn.modules import Module
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs


if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group


class DistributedDataParallelPythonBuckets(Module):
    """
    This is a slightly modified copy of the Pytorch code,
    which will be replaced when the upstream is fixed. You can see
    the upstream documentation
    `here <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_
    """

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        check_reduction=False,
        find_unused_parameters=False,
    ):

        super(DistributedDataParallelPythonBuckets, self).__init__()

        # Use all devices by default
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if output_device is None:
            output_device = device_ids[0]

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.broadcast_buffers = broadcast_buffers
        self.check_reduction = check_reduction

        MB = 1024 * 1024

        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 250 * MB

        # reduction bucket size
        self.bucket_bytes_cap = bucket_cap_mb * MB

        # Sync params and buffers
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states, self.broadcast_bucket_size)

        self._ddp_init_helper()

    def _ddp_init_helper(self):
        """
        Initialization helper function that does the following:

        (1) replicating the module from device[0] to the other devices
        (2) bucketing the parameters for reductions
        (3) resetting the bucketing states
        (4) registering the grad hooks
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        if len(self.device_ids) > 1:
            # TODO: we don't need to replicate params in here. they're always going to
            # be broadcasted using larger blocks in broadcast_coalesced, so it might be
            # better to not pollute the caches with these small blocks
            self._module_copies = replicate(self.module, self.device_ids, detach=True)
            self._module_copies[0] = self.module

            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(
                    self.module.parameters(), module_copy.parameters()
                ):
                    copy_param.requires_grad = param.requires_grad

        else:
            self._module_copies = [self.module]

        self.modules_params = [list(m.parameters()) for m in self._module_copies]
        self.modules_buffers = [list(m.buffers()) for m in self._module_copies]

        # This is a triply-nested list where the "dimensions" are: devices, buckets, bucket_elems
        param_buckets = []

        # Split the parameters into buckets and by types as well
        # We only need to bucket and reduce parameters that require grad and
        # this is also true for backward since only the backward hooks for
        # parameters that require grad will be registered with gradient
        # reduction functions
        params_to_bucket = [[] for _ in self._module_copies]
        for dev_idx, m in enumerate(self._module_copies):
            for p in m.parameters():
                if p.requires_grad:
                    params_to_bucket[dev_idx].append(p)

        param_buckets = [
            dist._dist_bucket_tensors(
                dev_params_to_bucket, int(self.bucket_bytes_cap), fine_grained=False
            )
            for dev_params_to_bucket in params_to_bucket
        ]

        self.bucket_sizes = []
        self.bucket_map = {}

        # We transpose param_buckets, so the loop is over buckets.
        # param_buckets_tuple is a doubly-nested list with "dims": devices, bucket_elems
        for bucket_idx, param_buckets_tuple in enumerate(zip(*param_buckets)):
            self.bucket_sizes.append(0)
            # Now, we transpose again, so we iterate over bucket_elems, but getting tuples
            # of params from each device.
            for param_tuple in zip(*param_buckets_tuple):
                if not param_tuple[0].requires_grad:
                    continue
                for p in param_tuple:
                    self.bucket_map[p] = (bucket_idx, self.bucket_sizes[bucket_idx])
                self.bucket_sizes[bucket_idx] += 1

        self.buckets = [
            [
                [None for _ in range(self.bucket_sizes[i])]
                for _ in range(len(self.device_ids))
            ]
            for i in range(len(self.bucket_sizes))
        ]
        # The number of params ready in each bucket
        self.buckets_ready_size = [
            [0 for _ in range(len(self.device_ids))]
            for i in range(len(self.bucket_sizes))
        ]

        # coalesced bucket for only device 0
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        # We will always reduce the bucket following the reverse order
        # that is, alway reduces following the order of: n - 1, n - 2, ..., 0
        self.next_bucket = len(self.bucket_sizes) - 1
        # When all buckets are reduced, this will be set to True. This flag is
        # useful for sanity checks to ensure that each iteration's backward has
        # always reduced all buckets
        self.all_buckets_reduced = False
        self.check_previous_reduction = False
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]
        self._register_grad_hooks()

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self._module_copies)

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs["process_group"], attrs["default_streams"], attrs["_grad_accs"]
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        self.check_previous_reduction = False
        super(DistributedDataParallelPythonBuckets, self).__setstate__(state)
        self._ddp_init_helper()

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True

        if pickle_not_supported:
            raise RuntimeError(
                "DDP Pickling/Unpickling are only supported "
                "when using DDP with the default process "
                "group. That is, when you have called "
                "init_process_group and have not passed "
                "process_group argument to DDP constructor"
            )

    def _check_previous_reduction(self):
        if not self.training:
            return
        # self.check_previous_reduction will be False in the first iteration
        # and is then toggled to True for all future iterations.
        if self.check_previous_reduction is False:
            self.check_previous_reduction = True
        else:
            if not self.all_buckets_reduced:
                raise RuntimeError(
                    "Not all gradients have been reduced from "
                    "the backward of the previous iteration. "
                    "This is an unexpected and fatal error. "
                    "Please check and ensure that the model's "
                    "parameters are not changed after you wrap "
                    "up the model with DistributedDataParallelPythonBuckets."
                )
        self.all_buckets_reduced = False

    def forward(self, *inputs, **kwargs):
        if self.check_reduction:
            self._check_previous_reduction()
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        self._sync_params()
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        outputs = self.parallel_apply(
            self._module_copies[: len(inputs)], inputs, kwargs
        )
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[: len(replicas)]
        )

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        self.check_previous_reduction = False
        super(DistributedDataParallelPythonBuckets, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        dist._dist_broadcast_coalesced(self.process_group, tensors, buffer_size, False)

    def _sync_params(self):
        with torch.no_grad():
            if len(self.device_ids) > 1:
                # intra-node parameter sync
                result = broadcast_coalesced(
                    self.modules_params[0], self.device_ids, self.broadcast_bucket_size
                )
                for tensors, module_params in zip(result[1:], self.modules_params[1:]):
                    for tensor, param in zip(tensors, module_params):
                        param.set_(tensor)

            # module buffer sync
            if self.broadcast_buffers and len(self.modules_buffers[0]) > 0:
                # cross-node buffer sync
                self._dist_broadcast_coalesced(
                    self.modules_buffers[0], self.broadcast_bucket_size
                )
                if len(self.device_ids) > 1:
                    # intra-node buffer sync
                    result = broadcast_coalesced(
                        self.modules_buffers[0],
                        self.device_ids,
                        self.broadcast_bucket_size,
                    )
                    for tensors, module_buffers in zip(
                        result[1:], self.modules_buffers[1:]
                    ):
                        for tensor, buffer in zip(tensors, module_buffers):
                            buffer.set_(tensor)

    def _passing_sync_batchnorm_handle(self, module_copies):
        for dev_idx, module in enumerate(module_copies):
            for layer in module.modules():
                if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                    layer._specify_ddp_gpu_num(len(self.device_ids))

    def _register_grad_hooks(self):
        self._grad_accs = []  # need to keep them in scope

        # default stream tracking to launch nccl reduce kernels
        self.default_streams = []
        for dev_id in self.device_ids:
            with torch.cuda.device(dev_id):
                self.default_streams.append(torch.cuda.current_stream())

        for device_idx, module in enumerate(self._module_copies):
            for p in module.parameters():
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(p, device_idx))
                    self._grad_accs.append(grad_acc)

    def _make_param_hook(self, param, device_idx):
        bucket_idx, bucket_offset = self.bucket_map[param]

        def distributed_data_parallel_hook(*unused):
            if param.grad.requires_grad:
                raise RuntimeError(
                    "DistributedDataParallelPythonBuckets only works "
                    "with gradients that don't require grad"
                )
            bucket = self.buckets[bucket_idx][device_idx]
            bucket[bucket_offset] = param.grad.data
            self.buckets_ready_size[bucket_idx][device_idx] += 1

            # We can flush these and save memory for replicas
            if device_idx > 0:
                param.grad = None
                with torch.no_grad():
                    param.set_()

            # Current device's bucket is full
            if (
                self.buckets_ready_size[bucket_idx][device_idx]
                == self.bucket_sizes[bucket_idx]
            ):
                self.devs_ready[bucket_idx] += 1
                if self.devs_ready[bucket_idx] < len(self.device_ids):
                    return

                # Now all devices's buckets with index: bucket_idx are ready
                if bucket_idx == self.next_bucket:
                    self._queue_reduction(bucket_idx)
                    self.next_bucket -= 1
                    # Now reduce anything that is ready but not yet reduced
                    if len(self.ready_buckets_not_reduced) > 0:
                        sorted_todo = sorted(
                            self.ready_buckets_not_reduced, reverse=True
                        )
                        for i in sorted_todo:
                            # Nothing can be reduced now
                            if i < self.next_bucket:
                                break
                            self._queue_reduction(i)
                            self.ready_buckets_not_reduced.remove(i)
                            if i == self.next_bucket:
                                self.next_bucket -= 1
                else:
                    self.ready_buckets_not_reduced.add(bucket_idx)

                # When all devices' buckets
                if self.next_bucket == -1:
                    # A final sync for all the reduction works
                    self._sync_reduction_works()
                    self.all_buckets_reduced = True

        return distributed_data_parallel_hook

    def _queue_reduction(self, bucket_idx):
        # _queue_reduction will use a seperate CUDA stream to coalesce
        # the small tensors to achieve more parallelisms, before passing the
        # coalesced tensor into the c10d CUDA stream for reduction
        result = dist._queue_reduction(
            self.process_group, self.buckets[bucket_idx], self.device_ids
        )
        self.reduction_works[bucket_idx] = result[0]
        self.buckets_coalesced[bucket_idx] = result[1]

    def _sync_reduction_works(self):
        # Now only work on the first GPU of self.device_ids
        # _sync_reduction will use a seperate CUDA stream to uncoalesce
        # the coalesced tensors to achieve more parallelisms
        for bucket_idx, grads_batch in enumerate(self.buckets):
            dist._sync_reduction(
                self.reduction_works[bucket_idx],
                grads_batch[0],
                self.buckets_coalesced[bucket_idx],
            )

        # Reset the module states
        self.next_bucket = len(self.bucket_sizes) - 1
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        self.buckets = [
            [
                [None for _ in range(self.bucket_sizes[i])]
                for _ in range(len(self.device_ids))
            ]
            for i in range(len(self.bucket_sizes))
        ]
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        self.buckets_ready_size = [
            [0 for _ in range(len(self.device_ids))]
            for i in range(len(self.bucket_sizes))
        ]
