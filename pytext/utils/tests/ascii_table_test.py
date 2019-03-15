#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

from pytext.utils.ascii_table import ascii_table


class TestTables(unittest.TestCase):
    def test_simple(self):
        two_rows = [
            {"stage": "forward", "total": "0.011"},
            {"stage": "comput_loss", "total": "0.000"},
        ]
        in_table_form = ascii_table(two_rows)

        output = (
            "+-------------+-------+\n"
            "|     forward | 0.011 |\n"
            "| comput_loss | 0.000 |\n"
            "+-------------+-------+"
        )
        self.assertEqual(in_table_form, output)

    def test_with_headers(self):
        two_rows = [
            {"stage": "forward", "total": "0.011"},
            {"stage": "comput_loss", "total": "0.000"},
        ]

        headers = {"stage": "Stage", "total": "Total Time"}

        in_table_form = ascii_table(two_rows, headers)

        output = (
            "+-------------+------------+\n"
            "| Stage       | Total Time |\n"
            "+-------------+------------+\n"
            "|     forward |      0.011 |\n"
            "| comput_loss |      0.000 |\n"
            "+-------------+------------+"
        )
        self.assertEqual(in_table_form, output)

    def test_with_headers_and_footer(self):
        two_rows = [
            {"stage": "forward", "total": "0.011"},
            {"stage": "comput_loss", "total": "0.000"},
        ]

        headers = {"stage": "Stage", "total": "Total Time"}
        footers = {"stage": "This very long footer is necessary.", "total": "0.055"}

        in_table_form = ascii_table(two_rows, headers, footers)

        output = (
            "+-------------------------------------+------------+\n"
            "| Stage                               | Total Time |\n"
            "+-------------------------------------+------------+\n"
            "|                             forward |      0.011 |\n"
            "|                         comput_loss |      0.000 |\n"
            "+-------------------------------------+------------+\n"
            "| This very long footer is necessary. | 0.055      |\n"
            "+-------------------------------------+------------+"
        )
        self.assertEqual(in_table_form, output)
