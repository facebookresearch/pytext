#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import timeit

from pytext.common.ascii_table import ascii_table


class StageTime:
    def __init__(self):
        self.total = 0
        self.count = 0

    def incr(self, elapsed):
        self.total += elapsed
        self.count += 1

    @property
    def average(self):
        return self.total / self.count if self.count else 0


class StageTimer:
    """
    Reports each stage Total Time, Average Time and Count format in ascii_table.

    Example:
    [forward] total: 21.470, average: 0.031, count: 702
    [add_metric] total: 15.764, average: 0.022, count: 702
    [overall] total: 38.076, average: 38.076, count: 1
    """

    def __init__(self):
        self.reset()

    def add_stage(self, stage):
        current = timeit.default_timer()
        # increment stage total time
        if stage not in self.per_stage_time:
            self.per_stage_time[stage] = StageTime()
        self.per_stage_time[stage].incr(current - self.checkpoint)

        self.checkpoint = current

    def report(self, header):
        total = timeit.default_timer() - self.start
        print(f"\n\t {header}")
        print(
            ascii_table(
                [
                    {
                        "stage": stage,
                        "total": f"{stage_time.total:.3f}",
                        "average": f"{stage_time.average:.3f}",
                        "count": f"{stage_time.count}",
                    }
                    for stage, stage_time in self.per_stage_time.items()
                ],
                human_column_names={
                    "stage": "Stage",
                    "total": "Total Time",
                    "average": "Average Time",
                    "count": "Count",
                },
                footer={
                    "stage": "Overall training",
                    "total": f"{total:.3f}",
                    "average": f"{total:.3f}",
                    "count": "1",
                },
                indentation="\t",
            )
        )

    def reset(self):
        self.start = timeit.default_timer()
        self.checkpoint = self.start
        self.per_stage_time = {}


@contextlib.contextmanager
def time_context(header):
    timer = StageTimer()
    try:
        yield timer
    finally:
        timer.report(header)
