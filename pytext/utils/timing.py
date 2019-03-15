#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import functools
import timeit
import traceback
import weakref

from .ascii_table import ascii_table


class SnapshotList(list):
    """lists are not weakref-able by default."""


class Timings:
    sum: float
    count: int
    max: float

    def __init__(self, sum: float = 0.0, count: int = 0, max: float = -float("inf")):
        self.sum = sum
        self.count = count
        self.max = max

    @property
    def average(self):
        return self.sum / (self.count or 1)

    def add(self, time):
        self.sum += time
        self.count += 1
        self.max = max(self.max, time)


SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR


def format_time(seconds):
    if seconds > 60:
        days, seconds = int(seconds // SECONDS_IN_DAY), seconds % SECONDS_IN_DAY
        hours, seconds = int(seconds // SECONDS_IN_HOUR), seconds % SECONDS_IN_HOUR
        minutes, seconds = (
            int(seconds // SECONDS_IN_MINUTE),
            seconds % SECONDS_IN_MINUTE,
        )
        if days:
            if minutes >= 30:
                hours += 1
            return f"{days}d{hours}h"
        elif hours:
            if seconds >= 30:
                minutes += 1
            return f"{hours}h{minutes}m"
        else:
            seconds = int(round(seconds))
            return f"{minutes}m{seconds}s"
    elif seconds > 1:
        return f"{seconds:.1f}s"
    elif seconds > 0.001:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds * 1000000:.1f}ns"


class Snapshot:
    def __init__(self):
        self.times = collections.defaultdict(Timings)
        self.start = timeit.default_timer()

    def report(self):
        snapshot_total = timeit.default_timer() - self.start

        def path(key):
            return " -> ".join(label for label, _ in key)

        results = [
            {
                "name": path(key),
                "total": format_time(times.sum),
                "avg": format_time(times.average),
                "max": format_time(times.max),
                "count": times.count,
            }
            for key, times in sorted(self.times.items())
        ]
        print(
            ascii_table(
                results,
                human_column_names={
                    "name": "Stage",
                    "total": "Total",
                    "avg": "Average",
                    "max": "Max",
                    "count": "Count",
                },
                footer={"name": "Total time", "total": format_time(snapshot_total)},
                alignments={"name": "<"},
            )
        )


class HierarchicalTimer:
    def __init__(self):
        self.current_stack = []
        self.all_snapshots = SnapshotList()

    def snapshot(self):
        snapshot = Snapshot()
        self.all_snapshots.append(weakref.ref(snapshot))
        return snapshot

    def _clean_snapshots(self):
        self.all_snapshots = [ref for ref in self.all_snapshots if ref() is not None]

    def push(self, label, caller_id):
        self.current_stack.append((label, caller_id, timeit.default_timer()))

    def pop(self):
        label, _, start_time = self.current_stack[-1]
        key = tuple((label, caller) for label, caller, _ in self.current_stack)
        delta = timeit.default_timer() - start_time
        for ref in self.all_snapshots:
            snapshot = ref()
            if snapshot is not None:
                snapshot.times[key].add(delta)
        self.current_stack.pop()
        # Need to put this somewhere
        self._clean_snapshots()

    def time(self, label):
        return _TimerContextManager(label, self)


class _TimerContextManager:
    def __init__(self, label, timer, caller_id=None):
        self.label = label
        self.timer = timer
        self.caller_id = caller_id

    def __enter__(self):
        if self.caller_id:
            caller_id = self.caller_id
        else:
            stack = traceback.extract_stack()
            caller = stack[-2]
            caller_id = (caller.filename, caller.line)
        self.timer.push(self.label, caller_id)

    def __exit__(self, *exception_info):
        self.timer.pop()

    def __call__(self, fn):
        """Decorator syntax"""
        caller_id = (fn.__code__.co_filename, fn.__code__.co_firstlineno)
        timer_context = _TimerContextManager(self.label, self.timer, caller_id)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with timer_context:
                return fn(*args, **kwargs)

        return wrapper


TIMER = HierarchicalTimer()


time = TIMER.time
snapshot = TIMER.snapshot
SNAPSHOT = TIMER.snapshot()
report = SNAPSHOT.report


def report_snapshot(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        snapshot = TIMER.snapshot()
        result = fn(*args, **kwargs)
        snapshot.report()
        return result

    return wrapper
