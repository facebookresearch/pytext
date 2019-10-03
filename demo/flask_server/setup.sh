#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p ~/miniconda
rm -f miniconda.sh
source miniconda/bin/activate

conda install -y protobuf
conda install -y boto3 flask future numpy pip
conda install -y pytorch -c pytorch

sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to 10000
