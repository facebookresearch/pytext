# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

CPPFLAGS += -g -std=c++11 -std=c++14 \
	-I./gen-cpp \
	-I/libtorch/include \
	-Wno-deprecated-declarations
CLIENT_LDFLAGS += -lthrift
SERVER_LDFLAGS += -L/libtorch/lib \
	-lthrift -lpistache -lpthread -ltorch -lc10 -lcurl -lgflags -lglog \
	-L/usr/local/lib -lsentencepiece -lsentencepiece_train -lprotobuf

server: server.o gen-cpp/Predictor.o
	g++ $^ $(SERVER_LDFLAGS) -o $@

clean:
	rm -f *.o server
