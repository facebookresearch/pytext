#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.file_io import HTTPURLHandler, PathManager


def register_http_url_handler():
    """
    support reading file from url starting with "http://", "https://", "ftp://"
    """
    PathManager.register_handler(HTTPURLHandler(), allow_override=True)
