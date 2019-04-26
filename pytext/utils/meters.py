#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time


class Meter:
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init):
        raise NotImplementedError

    def update(self, val=1):
        raise NotImplementedError

    @property
    def avg(self):
        return 0


class TimeMeter(Meter):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        super().__init__(init=init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)
