# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

CPPFLAGS += -g -std=c++11 -std=c++14 \
  -I./gen-cpp \
  -I/pytorch -I/pytorch/build \
	-I/pytorch/aten/src/ \
	-I/pytorch/third_party/protobuf/src/
CLIENT_LDFLAGS += -lthrift
SERVER_LDFLAGS += -L/pytorch/build/lib \
  -lthrift -lpistache -lpthread -lcaffe2 -lprotobuf -lc10 -lcurl

server: server.o gen-cpp/Predictor.o
	g++ $^ $(SERVER_LDFLAGS) -o $@

clean:
	rm -f *.o server
