#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
from typing import Any, Dict


def rows_to_columnar(rows) -> Dict[str, Any]:
    columnar = defaultdict(list)
    for row in rows:
        for column, value in row.items():
            columnar[column].append(value)
    return columnar


def columnar_to_rows(columnar):
    rows = []
    for index in range(len(columnar.values()[0])):
        row = {column: lists[index] for column, lists in columnar.items()}
        rows.append(row)
    return rows


def rows_to_columnar_tuple(rows):
    columnar_tuple = tuple([] for _ in range(len(rows[0])))
    for row in rows:
        for i, item in enumerate(row):
            columnar_tuple[i].append(item)
    return columnar_tuple


def columnar_tuple_to_rows(columnar_tuple):
    rows = []
    for index in range(len(columnar_tuple[0])):
        row = [lists[index] for lists in columnar_tuple if lists]
        rows.append(row)
    return rows
