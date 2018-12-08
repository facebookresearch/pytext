#! /bin/bash
sudo apt-get update
sudo apt-get install -y cmake python-pip python-dev build-essential protobuf-compiler libprotoc-dev
sudo ./install_deps
sudo pip install --progress-bar off http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
sudo pip install --progress-bar off pytest pytest-cov