# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

FROM ubuntu:16.04

# Install Caffe2 + dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libgoogle-glog-dev \
  libgtest-dev \
  libiomp-dev \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libopenmpi-dev \
  libsnappy-dev \
  openmpi-bin \
  openmpi-doc \
  python-dev \
  python-pip
RUN pip install --upgrade pip
RUN pip install setuptools wheel
RUN pip install future numpy protobuf typing hypothesis pyyaml
RUN apt-get install -y --no-install-recommends \
      libgflags-dev \
      cmake
RUN git clone https://github.com/pytorch/pytorch.git
WORKDIR pytorch
RUN git submodule update --init --recursive
RUN python setup.py install

# Install Thrift + dependencies
WORKDIR /
RUN apt-get update && apt-get install -y \
  libboost-dev \
  libboost-test-dev \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-thread-dev \
  libevent-dev \
  automake \
  libtool \
  curl \
  flex \
  bison \
  pkg-config \
  libssl-dev
RUN curl https://www-us.apache.org/dist/thrift/0.11.0/thrift-0.11.0.tar.gz --output thrift-0.11.0.tar.gz
RUN tar -xvf thrift-0.11.0.tar.gz
WORKDIR thrift-0.11.0
RUN ./bootstrap.sh
RUN ./configure
RUN make
RUN make install

# Install Pistache (REST framework)
WORKDIR /
RUN git clone https://github.com/oktal/pistache.git
WORKDIR /pistache
RUN git submodule update --init
RUN mkdir build
WORKDIR /pistache/build
RUN cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
RUN make
RUN make install

# Install libcurl
RUN apt-get install -y libcurl4-openssl-dev

# Copy local files to /app
COPY . /app
WORKDIR /app

# Compile app
RUN thrift -r --gen cpp predictor.thrift
RUN make

# Add library search paths
RUN echo '/pytorch/build/lib/' >> /etc/ld.so.conf.d/local.conf
RUN echo '/usr/local/lib/' >> /etc/ld.so.conf.d/local.conf
RUN ldconfig

# Open ports
EXPOSE 9090
EXPOSE 8080
