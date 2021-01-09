#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import math
import os

# keep PathManager here for more flexibility until PathManager becomes more mature
# in case we want some hacks in PathManager, we can do it here without updating
# the import everywhere in PyText
# TODO: @stevenliu use PathManagerFactory after it's released to PyPI
from iopath.common.file_io import HTTPURLHandler, PathManager as PathManagerBase


PathManager = PathManagerBase()


def register_http_url_handler():
    """
    support reading file from url starting with "http://", "https://", "ftp://"
    """
    PathManager.register_handler(HTTPURLHandler(), allow_override=True)


def chunk_file(file_path, chunks, work_dir):
    """Splits a large file by line into number of chunks and writes them into work_dir"""
    with PathManager.open(file_path) as fin:
        num_lines = sum(1 for line in fin)

    chunk_size = math.ceil(num_lines / chunks)
    output_file_paths = []
    with contextlib.ExitStack() as stack:
        fin = stack.enter_context(PathManager.open(file_path))
        for i, line in enumerate(fin):
            if not i % chunk_size:
                file_split = "{}.chunk_{}".format(
                    os.path.join(work_dir, os.path.basename(file_path)), i // chunk_size
                )
                output_file_paths.append(file_split)
                fout = stack.enter_context(open(file_split, "w"))
            fout.write("{}\n".format(line.strip()))

    return output_file_paths
