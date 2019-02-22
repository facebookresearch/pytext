#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
sudo apt-get update
sudo apt-get install -y cmake python-pip python-dev build-essential protobuf-compiler libprotoc-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo ./install_deps
sudo pip install --progress-bar off pytest pytest-cov
