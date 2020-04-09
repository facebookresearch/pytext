Serve Models in Production
======================================================

We have seen how to use PyText models in an app using Flask in the `previous tutorial <pytext_models_in_your_app.html>`_, but the server implementation still requires a Python runtime. TorchScript models are designed to perform well even in production scenarios with high requirements for performance and scalability.

In this tutorial, we will implement a Thrift server in C++, in order to extract the maximum performance from our exported TorchScript text classification model trained on the demo dataset. We will also prepare a Docker image which can be deployed to your cloud provider of choice.

The full source code for the implemented server in this tutorial can be found in the `demos directory <https://github.com/facebookresearch/pytext/tree/master/demo/predictor_service>`_.

To complete this tutorial, you will need to have `Docker <https://www.docker.com/products/docker-desktop>`_ installed.

1. Create a Dockerfile and install dependencies
------------------------------------------------------

The first step is to prepare our Docker image with the necessary dependencies (including `libtorch <https://pytorch.org/cppdocs/installing.html>`_ for running TorchScript models). In an empty, folder, create a *Dockerfile* with the `following contents <https://github.com/facebookresearch/pytext/tree/master/demo/predictor_service/Dockerfile>`_:

**Dockerfile**

.. code-block:: dockerfile

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

  # Install libtorch
  WORKDIR /
  RUN curl https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip --output libtorch.zip \
      && unzip libtorch.zip \
      && rm libtorch.zip

2. Add Thrift API
---------------------------------------------

`Thrift <https://thrift.apache.org/>`_ is a software library for developing scalable cross-language services. It comes with a client code generation engine enabling services to be interfaced across the network on multiple languages or devices. We will use Thrift to create a service which serves our model.

Add the dependency on Thrift in the Dockerfile:

.. code-block:: dockerfile

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

Our C++ server will expose a very simple API that receives a sentence/utterance as a string, and return a map of label names(`string`) -> scores(`double`). The corresponding thrift spec fo the API is below:

**predictor.thrift**

.. code-block:: thrift

  namespace cpp predictor_service

  service Predictor {
    // Returns scores for each class
    map<string,double> predict(1:string doc),
  }

3. Implement server code
--------------------------

Now, we will write our server's code. The first thing our server needs to be able to do is to load the model from a file path into the Caffe2 workspace and initialize it. We do that in the constructor of our ``PredictorHandler`` thrift server class:

**server.cpp**

.. code-block:: cpp

  class PredictorHandler : virtual public PredictorIf {
    private:
      torch::jit::script::Module mModule;
  ...
    public:
      PredictorHandler(string& modelFile) {
        mModule = torch::jit::load(modelFile);
      }
  ...
  }


Now that our model is loaded, we need to implement the `predict` API method which is our main interface to clients. The implementation needs to do the following:

1. Pre-process the input sentence into tokens
2. Prepare input for the model as a batch
3. Run the model
4. Extract and populate the results into the response

**server.cpp**

.. code-block:: cpp

  class PredictorHandler : virtual public PredictorIf {
  ...
    public:
      void predict(map<string, double>& _return, const string& doc) {
        // Pre-process: tokenize input doc
        vector<string> tokens;
        string docCopy = doc;
        tokenize(tokens, docCopy);

        // Prepare input for the model as a batch
        vector<vector<string>> batch{tokens};
        vector<torch::jit::IValue> inputs{
            mDummyVec, // texts in model.forward
            mDummyVecVec, // multi_texts in model.forward
            batch // tokens in model.forward
        };

        // Run the model
        auto output =
            mModule.forward(inputs).toGenericListRef().at(0).toGenericDict();

        // Extract and populate results into the response
        for (const auto& elem : output) {
          _return.insert({elem.key().toStringRef(), elem.value().toDouble()});
        }
      }
  ...
  }

The full source code for *server.cpp* can be found `here <https://github.com/facebookresearch/pytext/tree/master/demo/predictor_service/server.cpp>`__.

Note: The source code in the demo also implements a REST proxy for the Thrift server to make it easy to test and make calls over simple HTTP. Feel free to use that if you don't need to pass raw tensors into your model.

4. Build and compile scripts
------------------------------

To build our server, we need to provide necessary headers during compile time and the required dependent libraries during link time. The *Makefile* below does this:

**Makefile**

.. code-block:: Makefile

  CPPFLAGS += -g -std=c++11 -std=c++14 \
    -I./gen-cpp \
    -I/libtorch/include \
    -Wno-deprecated-declarations
  CLIENT_LDFLAGS += -lthrift
  SERVER_LDFLAGS += -L/libtorch/lib \
    -lthrift -lpistache -lpthread -ltorch -lc10 -lcurl -lgflags

  server: server.o gen-cpp/Predictor.o
    g++ $^ $(SERVER_LDFLAGS) -o $@

  clean:
    rm -f *.o server

In our *Dockerfile*, we also add some steps to copy our local files into the docker image, compile the app, and add the necessary library search paths.

**Dockerfile**

.. code-block:: Dockerfile

  # Copy local files to /app
  COPY . /app
  WORKDIR /app

  # Compile app
  RUN thrift -r --gen cpp predictor.thrift
  RUN make

  # Add library search paths
  ENV LD_LIBRARY_PATH /libtorch/lib:/usr/local/lib

5. Test/Run the server
-------------------------

To obtain a sample TorchScript model, run the following commads in your PyText directory:

.. code-block:: console

  (pytext) $ pytext train < demo/configs/docnn.json
  (pytext) $ pytext torchscript-export < demo/configs/docnn.json

This creates a */tmp/model.pt.torchscript* file which you should copy into the server directory where you wrote the files in the previous section. This section assumes that this directory matches the one found `here <https://github.com/facebookresearch/pytext/blob/master/demo/predictor_service/>`__. 

1. Build the Docker image:

.. code-block:: console

  $ docker build -t predictor_service .

If successful, you should see the message "Successfully tagged predictor_service:latest".

2. Run the server:

.. code-block:: console

  $ docker run -it -p 8080:8080 predictor_service:latest ./server model.pt.torchscript

If successful, you should see the message "Server running. Thrift port: 9090, REST port: 8080"

3. Test our server by sending a test utterance "set an alarm":

.. code-block:: console

  $ curl -G "http://localhost:8080" --data-urlencode "doc=set an alarm"

If successful, you should see the scores printed out on the console. On further inspection, the score for "alarm/set_alarm" is the highest among the classes. ::

  alarm/modify_alarm:-1.99205
  alarm/set_alarm:-1.8802
  alarm/snooze_alarm:-1.89931
  alarm/time_left_on_alarm:-2.00953
  reminder/set_reminder:-2.00718
  reminder/show_reminders:-1.91181
  weather/find:-1.93019

Congratulations! You have now built your own server that can serve your PyText models in production!

We also provide a `Docker image on Docker Hub <https://hub.docker.com/r/pytext/predictor_service>`_ with this example, which you can freely use and adapt to your needs.
