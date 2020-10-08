#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.metrics.calibration_metrics import (
    calculate_error,
    get_bucket_accuracy,
    get_bucket_confidence,
    get_bucket_scores,
)


class CalibrationUtilsTest(unittest.TestCase):
    def test_calibration(self):
        buckets = 10
        conf_list = [0.84, 0.98, 0.97, 0.76, 0.59, 0.62, 0.40, 0.33, 0.54, 0.37]
        true_list = [1, 5, 3, 2, 4, 2, 7, 4, 5, 2]
        pred_list = [1, 5, 3, 5, 2, 1, 7, 4, 5, 3]

        bucket_values, bucket_indices = get_bucket_scores(conf_list, buckets)
        bucket_confidence = get_bucket_confidence(bucket_values)
        bucket_accuracy = get_bucket_accuracy(bucket_indices, true_list, pred_list)
        expected_error, max_error, total_error = calculate_error(
            len(conf_list), bucket_values, bucket_confidence, bucket_accuracy
        )

        self.assertEqual(
            bucket_values,
            [
                [],
                [],
                [],
                [0.33, 0.37],
                [0.4],
                [0.59, 0.54],
                [0.62],
                [0.76],
                [0.84],
                [0.98, 0.97],
            ],
        )
        self.assertEqual(
            bucket_indices, [[], [], [], [7, 9], [6], [4, 8], [5], [3], [0], [1, 2]]
        )
        self.assertEqual(
            bucket_confidence,
            [-1.0, -1.0, -1.0, 0.35, 0.4, 0.565, 0.62, 0.76, 0.84, 0.975],
        )
        self.assertEqual(
            bucket_accuracy, [-1.0, -1.0, -1.0, 0.5, 1.0, 0.5, 0.0, 0.0, 1.0, 1.0]
        )
        self.assertAlmostEqual(expected_error, 26.2)
        self.assertAlmostEqual(max_error, 76.0)
        self.assertAlmostEqual(total_error, 238.0)
