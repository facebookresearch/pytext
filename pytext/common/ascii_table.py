#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools


def ascii_table(data, human_column_names=None, footer=None, indentation=""):
    data = list(data)
    columns = human_column_names or set(itertools.chain.from_iterable(data))
    widths = {
        column: max(len(str(row.get(column))) for row in data) for column in columns
    }
    if human_column_names:
        for column, human in human_column_names.items():
            widths[column] = max(widths[column], len(human))

    separator = "+" + "+".join("-" * (width + 2) for width in widths.values()) + "+"

    def format_row(row, alignment=">"):
        return (
            "| "
            + " | ".join(
                format(row.get(column, ""), f"{alignment}{width}")
                for column, width in widths.items()
            )
            + " |"
        )

    header = (
        (format_row(human_column_names, alignment="<"), separator)
        if human_column_names
        else ()
    )

    footer = (format_row(footer, alignment="<"), separator) if footer else ()

    return indentation + f"\n{indentation}".join(
        (separator, *header, *(format_row(row) for row in data), separator, *footer)
    )


def ascii_table_from_dict(dict, key_name, value_name, indentation=""):
    return ascii_table(
        [{"key": key, "value": value} for key, value in dict.items()],
        {"key": key_name, "value": value_name},
        indentation=indentation,
    )
