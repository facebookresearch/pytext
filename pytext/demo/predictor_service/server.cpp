// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include "gen-cpp/Predictor.h"

#include "pistache/endpoint.h"

#include <curl/curl.h>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/predictor/predictor_utils.h"
#include "caffe2/proto/predictor_consts.pb.h"
#include "caffe2/utils/proto_utils.h"

using namespace std;

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;
using namespace predictor_service;

using namespace Pistache;

using namespace caffe2;
using namespace db;
using namespace predictor_utils;

// Main handler for the predictor service
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

    void tokenize(vector<string>& tokens, string& doc) {
      transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
      int start = 0;
      int end = 0;
      for (int i = 0; i < doc.length(); i++) {
        if (isspace(doc.at(i))){
          end = i;
          if (end != start) {
            tokens.push_back(doc.substr(start, end - start));
          }

          start = i + 1;
        }
      }

      if (start < doc.length()) {
        tokens.push_back(doc.substr(start, doc.length() - start));
      }

      if (tokens.size() == 0) {
        // Add PAD_TOKEN in case of empty text
        tokens.push_back("<pad>");
      }
    }

  public:
    PredictorHandler(string &modelFile): mWorkspace("workspace") {
      mPredictNet = loadAndInitModel(mWorkspace, modelFile);
    }

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
};

// REST proxy for the predictor Thrift service (not covered in tutorial)
class RestProxyHandler : public Http::Handler {
  private:
    shared_ptr<TTransport> mTransport;
    shared_ptr<PredictorClient> mPredictorClient;

    string urlDecode(const string &encoded)
    {
      CURL *curl = curl_easy_init();
      int decodedLength;
      char *decodedCstr = curl_easy_unescape(
        curl, encoded.c_str(), encoded.length(), &decodedLength);
      string decoded(decodedCstr, decodedCstr + decodedLength);
      curl_free(decodedCstr);
      curl_easy_cleanup(curl);
      return decoded;
    }

  public:
    HTTP_PROTOTYPE(RestProxyHandler)

    RestProxyHandler(
      shared_ptr<TTransport>& transport,
      shared_ptr<PredictorClient>& predictorClient
    ) {
      mTransport = transport;
      mPredictorClient = predictorClient;
    }

    void onRequest(const Http::Request& request, Http::ResponseWriter response) {
      const string docParam = "doc";
      if (!mTransport->isOpen()) {
        mTransport->open();
      }

      stringstream out;
      if (request.query().has(docParam)) {
        string doc = urlDecode(request.query().get(docParam).get());
        map<string, vector<double>> labelScores;
        mPredictorClient->predict(labelScores, doc);
        map<string, vector<double>>::iterator it;
        for (it = labelScores.begin(); it != labelScores.end(); it++) {
          out << it->first << ":";
          for (int i = 0; i < it->second.size(); i++) {
            out << it->second.at(i) << " ";
          }

          out << endl;
        }

        response.send(Http::Code::Ok, out.str());
      }
      else {
        out << "Missing query parameter: " << docParam << endl;
        response.send(Http::Code::Bad_Request, out.str());
      }
    }
};

int main(int argc, char **argv) {
  // Parse command line args
  if (argc < 2) {
    cerr << "Usage:" << endl;
    cerr << "./server <miniDB file>" << endl;
    cerr << "./server <miniDB file> --no-rest" << endl;
    return 1;
  }

  string modelFile = argv[1];
  bool proxyEnabled = true;
  if (argc >= 3) {
    const string noRestOption = "--no-rest";
    if (noRestOption.compare(argv[2]) == 0) {
      proxyEnabled = false;
    }
    else {
      cerr << "Unrecognized argument: " << argv[2] << endl;
      return 1;
    }
  }

  int thriftPort = 9090;
  int restPort = 8080;

  // Initialize predictor thrift service
  shared_ptr<PredictorHandler> handler(new PredictorHandler(modelFile));
  shared_ptr<TProcessor> processor(new PredictorProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(thriftPort));
  shared_ptr<TTransportFactory> transportFactory(
    new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  TSimpleServer thriftServer(
    processor, serverTransport, transportFactory, protocolFactory);
  thread thriftThread([&](){ thriftServer.serve(); });
  cout << "Server running. Thrift port: " << thriftPort;

  if (proxyEnabled) {
    // Initialize Thrift client used to foward requests from REST
    shared_ptr<TTransport> socket(new TSocket("127.0.0.1", 9090));
    shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    shared_ptr<PredictorClient> predictorClient(new PredictorClient(protocol));

    // Initialize REST proxy
    Address addr(Ipv4::any(), Port(restPort));
    auto opts = Http::Endpoint::options().threads(1);
    Http::Endpoint restServer(addr);
    restServer.init(opts);
    restServer.setHandler(
      make_shared<RestProxyHandler>(transport, predictorClient));
    thread restThread([&](){ restServer.serve(); });

    cout << ", REST port: " << restPort << endl;
    restThread.join();
  }

  cout << endl;
  thriftThread.join();
  return 0;
}
