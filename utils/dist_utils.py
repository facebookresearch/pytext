#!/usr/bin/env python3

import os
import signal

import torch
import torch.distributed as dist_c10d


def dist_init(
    distributed_rank: int, world_size: int, init_method: str, backend: str = "nccl"
):
    if init_method and world_size > 1 and torch.cuda.is_available():
        dist_c10d.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=distributed_rank,
        )

        if distributed_rank != 0:
            suppress_output()


def suppress_output():
    import builtins as __builtin__

    def print(*args, **kwargs):
        pass

    __builtin__.print = print


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        import threading

        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)
