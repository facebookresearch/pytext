#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict


def rows_to_columnar(rows):
    columnar = defaultdict(list)
    for row in rows:
        for column, value in row.items():
            columnar[column].append(value)
    return columnar


def columnar_to_rows(columnar):
    rows = []
    for index in range(len(columnar.values()[0])):
        for column, lists in columnar.items():
            rows.append({column: lists[index]})
    return rows
