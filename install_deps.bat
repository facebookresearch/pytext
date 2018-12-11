@ECHO OFF
::Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
python -m pip install --upgrade pip
pip install -e . --process-dependency-links --no-cache-dir --progress-bar off
