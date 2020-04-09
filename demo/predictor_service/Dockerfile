# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

FROM ubuntu:18.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  libcurl4-openssl-dev \
  libgflags-dev \
  unzip

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
    flex \
    bison \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://downloads.apache.org/thrift/0.13.0/thrift-0.13.0.tar.gz --output thrift-0.13.0.tar.gz \
    && tar -xvf thrift-0.13.0.tar.gz \
    && rm thrift-0.13.0.tar.gz
WORKDIR /thrift-0.13.0
RUN ./bootstrap.sh \
    && ./configure \
    && make \
    && make install
    
# Install Pistache (C++ REST framework)
WORKDIR /
RUN git clone https://github.com/oktal/pistache.git
WORKDIR /pistache
RUN git submodule update --init \
    && mkdir build
WORKDIR /pistache/build
RUN cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .. \
    && make \
    && make install

# Install libtorch
WORKDIR /
RUN curl https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip --output libtorch.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip

# Copy local files to /app
COPY . /app
WORKDIR /app

# Compile app
RUN thrift -r --gen cpp predictor.thrift
RUN make

# Add library search paths
ENV LD_LIBRARY_PATH /libtorch/lib:/usr/local/lib

# Expose ports for Thrift and REST
EXPOSE 9090
EXPOSE 8080
