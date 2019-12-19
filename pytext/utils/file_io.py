#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
TODO: @stevenliu Deprecate this file after borc available in PyPI
"""
import os
import shutil
from typing import List


try:  # noqa
    from fvcore.common.file_io import PathManager

except ImportError:

    class PathManager:
        @staticmethod
        def open(*args, **kwargs):
            return open(*args, **kwargs)

        @staticmethod
        def copy(*args, **kwargs) -> bool:
            try:
                shutil.copyfile(*args, **kwargs)
                return True
            except Exception as e:
                print("Error in file copy - {}".format(str(e)))
                return False

        @staticmethod
        def get_local_path(path: str) -> str:
            return path

        @staticmethod
        def exists(path: str) -> bool:
            return os.path.exists(path)

        @staticmethod
        def isfile(path: str) -> bool:
            return os.path.isfile(path)

        @staticmethod
        def isdir(path: str) -> bool:
            return os.path.isdir(path)

        @staticmethod
        def ls(path: str) -> List[str]:
            return os.listdir(path)

        @staticmethod
        def mkdirs(*args, **kwargs):
            os.makedirs(*args, exist_ok=True, **kwargs)

        @staticmethod
        def rm(*args, **kwargs):
            os.remove(*args, **kwargs)
