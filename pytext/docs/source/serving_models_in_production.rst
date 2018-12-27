Serve Models in Production
======================================================

We have seen how to use PyText models in an app using Flask in the `previous tutorial <pytext_models_in_your_app.html>`_, but the server implementation still requires a Python runtime. Caffe2 models are designed to perform well even in production scenarios with high requirements for performance and scalability.

In this tutorial, we will implement a Thrift server in C++, in order to extract the maximum performance from our exported Caffe2 intent-slot model trained on the ATIS dataset. We will also prepare a Docker image which can be deployed to your cloud provider of choice.

The full source code for the implemented server in this tutorial can be found in the `demos directory <https://github.com/facebookresearch/pytext/tree/master/demo/predictor_service>`_.

To complete this tutorial, you will need to have `Docker <https://www.docker.com/products/docker-desktop>`_ installed.

1. Create a Dockerfile and install dependencies
------------------------------------------------------

The first step is to prepare our Docker image with the necessary dependencies. In an empty, folder, create a *Dockerfile* with the `following contents <https://github.com/facebookresearch/pytext/tree/master/demo/predictor_service/Dockerfile>`_:

**Dockerfile**

.. code-block:: dockerfile

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

2. Add Thrift API
---------------------------------------------

`Thrift <https://thrift.apache.org/>`_ is a software library for developing scalable cross-language services. It comes with a client code generation engine enabling services to be interfaced across the network on multiple languages or devices. We will use Thrift to create a service which serves our model.

Our C++ server will expose a very simple API that receives an sentence/utterance as a string, and return a map of label names(`string`) -> scores(`list<double>`). For document scores, the list will only contain one score, and for word scores, the list will contain one score per word. The corresponding thrift spec fo the API is below:

**predictor.thrift**

.. code-block:: thrift

  namespace cpp predictor_service

  service Predictor {
     // Returns list of scores for each label
     map<string,list<double>> predict(1:string doc),
  }

3. Implement server code
--------------------------

Now, we will write our server's code. The first thing our server needs to be able to do is to load the model from a file path into the Caffe2 workspace and initialize it. We do that in the constructor of our ``PredictorHandler`` thrift server class:

**server.cpp**

.. code-block:: cpp

  class PredictorHandler : virtual public PredictorIf {
    private:
      NetDef mPredictNet;
      Workspace mWorkspace;

      NetDef loadAndInitModel(Workspace& workspace, string& modelFile) {
        auto db = unique_ptr<DBReader>(new DBReader("minidb", modelFile));
        auto metaNetDef = runGlobalInitialization(move(db), &workspace);
        const auto predictInitNet = getNet(
          *metaNetDef.get(),
          PredictorConsts::default_instance().predict_init_net_type()
        );
        CAFFE_ENFORCE(workspace.RunNetOnce(predictInitNet));

        auto predictNet = NetDef(getNet(
          *metaNetDef.get(),
          PredictorConsts::default_instance().predict_net_type()
        ));
        CAFFE_ENFORCE(workspace.CreateNet(predictNet));

        return predictNet;
      }
  ...
    public:
      PredictorHandler(string &modelFile): mWorkspace("workspace") {
        mPredictNet = loadAndInitModel(mWorkspace, modelFile);
      }
  ...
  }


Now that our model is loaded, we need to implement the `predict` API method which is our main interface to clients. The implementation needs to do the following:

1. Pre-process the input sentence into tokens
2. Feed the input as tensors to the model
3. Run the model
4. Extract and populate the results into the response

**server.cpp**

.. code-block:: cpp

  class PredictorHandler : virtual public PredictorIf {
  ...
    public:
      void predict(map<string, vector<double>>& _return, const string& doc) {
        // Pre-process: tokenize input doc
        vector<string> tokens;
        string docCopy = doc;
        tokenize(tokens, docCopy);

        // Feed input to model as tensors
        Tensor valTensor = TensorCPUFromValues<string>(
          {static_cast<int64_t>(1), static_cast<int64_t>(tokens.size())}, {tokens}
        );
        BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_vals_str:value"), CPU)
          ->CopyFrom(valTensor);
        Tensor lensTensor = TensorCPUFromValues<int>(
          {static_cast<int64_t>(1)}, {static_cast<int>(tokens.size())}
        );
        BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_lens"), CPU)
          ->CopyFrom(lensTensor);

        // Run the model
        CAFFE_ENFORCE(mWorkspace.RunNet(mPredictNet.name()));

        // Extract and populate results into the response
        for (int i = 0; i < mPredictNet.external_output().size(); i++) {
          string label = mPredictNet.external_output()[i];
          _return[label] = vector<double>();
          Tensor scoresTensor = mWorkspace.GetBlob(label)->Get<Tensor>();
          for (int j = 0; j < scoresTensor.numel(); j++) {
            float score = scoresTensor.data<float>()[j];
            _return[label].push_back(score);
          }
        }
      }
  ...
  }

The full source code for *server.cpp* can be found `here <https://github.com/facebookresearch/pytext/tree/master/demo/predictor_service/server.cpp>`__.

Note: The source code in the demo also implements a REST proxy for the Thrift server to make it easy to test and make calls over simple HTTP, however it is not covered in the scope of this tutorial since the Thrift protocol is what we'll use in production.

4. Build and compile scripts
------------------------------

To build our server, we need to provide necessary headers during compile time and the required dependent libraries during link time: *libthrift.so*, *libcaffe2.so*, *libprotobuf.so* and *libc10.so*. The *Makefile* below does this:

**Makefile**

.. code-block:: Makefile

  CPPFLAGS += -g -std=c++11 -std=c++14 \
    -I./gen-cpp \
    -I/pytorch -I/pytorch/build \
  	-I/pytorch/aten/src/ \
  	-I/pytorch/third_party/protobuf/src/
  CLIENT_LDFLAGS += -lthrift
  SERVER_LDFLAGS += -L/pytorch/build/lib -lthrift -lcaffe2 -lprotobuf -lc10

  # ...

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
  RUN echo '/pytorch/build/lib/' >> /etc/ld.so.conf.d/local.conf
  RUN echo '/usr/local/lib/' >> /etc/ld.so.conf.d/local.conf
  RUN ldconfig

5. Test/Run the server
-------------------------

This section assumes that your local files match the one found `here <https://github.com/facebookresearch/pytext/tree/master/demo/predictor>`__.

Now that you have implemented your server, we will run the following commands to take it for a test run. In your server folder:

1. Build the image:

.. code-block:: console

  $ docker build -t predictor_service .

If successful, you should see the message "Successfully tagged predictor_service:latest".

2. Run the server. We use *models/atis_joint_model.c2* as the local path to our model file (add your trained model there):

.. code-block:: console

  $ docker run -it -p 8080:8080 predictor_service:latest ./server models/atis_joint_model.c2

If successful, you should see the message "Server running. Thrift port: 9090, REST port: 8080"

3. Test our server by sending a test utterance "Flight from Seattle to San Francisco":

.. code-block:: console

  $ curl -G "http://localhost:8080" --data-urlencode "doc=Flights from Seattle to San Francisco"

If successful, you should see the scores printed out on the console. On further inspection, the doc score for "flight", the 3rd word score for "B-fromloc.city_name" corresponding to "Seattle", the 5th word score for "B-toloc.city_name" corresponding to "San", and the 6th word score for "I-toloc.city_name" corresponding to "Francisco" should be close to 0. ::

  doc_scores:flight:-2.07426e-05
  word_scores:B-fromloc.city_name:-14.5363 -12.8977 -0.000172928 -12.9868 -9.94603 -16.0366
  word_scores:B-toloc.city_name:-15.2309 -15.9051 -9.89932 -12.077 -0.000134 -8.52712
  word_scores:I-toloc.city_name:-13.1989 -16.8094 -15.9375 -12.5332 -10.7318 -0.000501401

Congratulations! You have now built your own server that can serve your PyText models in production!

We also provide a `Docker image on Docker Hub <https://hub.docker.com/r/pytext/predictor_service>`_ with this example, which you can freely use and adapt to your needs.
