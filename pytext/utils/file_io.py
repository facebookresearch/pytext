#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# keep PathManager here for more flexibility until PathManager becomes more mature
# in case we want some hacks in PathManager, we can do it here without updating
# the import everywhere in PyText
from fvcore.common.file_io import HTTPURLHandler, PathManagerBase  # noqa


PathManager = PathManagerBase()


def register_http_url_handler():
    """
    support reading file from url starting with "http://", "https://", "ftp://"
    """
    PathManager.register_handler(HTTPURLHandler(), allow_override=True)
