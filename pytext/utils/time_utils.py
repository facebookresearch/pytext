#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib

from progressbar import Bar, Percentage, ProgressBar, Timer


class EpochProgressBar:
    """
    To make the progress bar independent of the total batches number, we use
    progress_scale to scale and make sure it only update progress_scale times.
    """

    def __init__(self, enable_bar, data_iter, rank, epoch, progress_scale):
        # data_iter type: BatchIterator
        self.enable_bar = enable_bar
        if self.enable_bar:
            self.total_batches = len(data_iter)
            self.bar = ProgressBar(
                maxval=self.total_batches,
                widgets=[
                    f"\nRank{rank} at epoch{epoch}: ",
                    Percentage(),
                    Bar(),
                    Timer(),
                ],
            )
            self.update_denom = max(self.total_batches // progress_scale, 1)

    def start(self):
        if self.enable_bar:
            self.bar.start()

    def finish(self):
        if self.enable_bar:
            self.bar.finish()

    def update(self, batch_id):
        if self.enable_bar and (
            batch_id % self.update_denom == 0 or batch_id == self.total_batches
        ):
            self.bar.update(batch_id)


@contextlib.contextmanager
def progress_bar(enabled, data_iter, rank, epoch, progress_scale=100):
    # data_iter type: BatchIterator
    bar = EpochProgressBar(enabled, data_iter, rank, epoch, progress_scale)
    try:
        bar.start()
        yield bar
    finally:
        bar.finish()
