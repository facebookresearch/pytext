#! /bin/bash
sudo apt-get update
sudo apt-get install -y cmake python-pip python-dev build-essential protobuf-compiler libprotoc-dev
sudo ./install_deps
sudo pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
