// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <csignal>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include "gen-cpp/Predictor.h"

#include "pistache/endpoint.h"
#include <sentencepiece_processor.h>
#include <curl/curl.h>
#include <torch/script.h>

#include "formatter.hpp"

#include <glog/logging.h>
#include <gflags/gflags.h>
DEFINE_int32(
    port_thrift,
    9090,
    "Port which the Thrift server should listen on");
DEFINE_bool(rest, true, "Set up a REST proxy to the Thrift server");
DEFINE_int32(port_rest, 8080, "Port which the REST proxy should listen on");
DEFINE_bool(run_tests, true, "Run tests before starting the server");

using namespace std;

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;
using namespace predictor_service;

using namespace Pistache;

// Server objects need to be global so that the handler can kill them cleanly
unique_ptr<TSimpleServer> thriftServer;
unique_ptr<Http::Endpoint> restServer;
void shutdownHandler(int s) {
  if (thriftServer) {
    LOG(INFO) << "Shutting down Thrift server";
    thriftServer->stop();
  }
  if (restServer) {
    LOG(INFO) << "Shutting down REST proxy server";
    restServer->shutdown();
  }
  exit(0);
}

// Main handler for the predictor service
class PredictorHandler : virtual public PredictorIf {
 private:
  torch::jit::script::Module mModule;
  sentencepiece::SentencePieceProcessor mSpProcessor;
  bool mUseSentencePiece;

  c10::optional<vector<string>> c10mDummyVec;
  c10::optional<vector<vector<string>>> c10mDummyVecVec;

  void sentencepieceTokenize(vector<string>& tokens, string& doc) {
    mSpProcessor.Encode(doc, &tokens);
  }

  void tokenize(vector<string>& tokens, string& doc) {
    size_t start = 0;
    size_t end = 0;
    for (size_t i = 0; i < doc.length(); i++) {
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
  // For XLM-R models
  PredictorHandler(string& modelFile, string& sentencepieceVocabFile) {
    mModule = torch::jit::load(modelFile);
    const auto status = mSpProcessor.Load(sentencepieceVocabFile);
    if (!status.ok()) {
      LOG(FATAL) << status.ToString();
    }

    LOG(INFO) << "Loaded SentencePiece model from " << sentencepieceVocabFile;
    mUseSentencePiece = true;
  }

  // For DocNN models
  PredictorHandler(string& modelFile) {
    mModule = torch::jit::load(modelFile);
    mUseSentencePiece = false;
  }

  void predict(map<string, double>& _return, const string& doc) {
    // Pre-process: tokenize input doc
    vector<string> tokens;
    string docCopy = doc;
    if (mUseSentencePiece) {
      sentencepieceTokenize(tokens, docCopy);
    } else {
      tokenize(tokens, docCopy);
    }

    if (VLOG_IS_ON(1)) {
      stringstream ss;
      ss << "[";
      copy(tokens.begin(), tokens.end(), ostream_iterator<string>(ss, ", "));
      ss.seekp(-1, ss.cur); ss << "]";
      VLOG(1) << "Tokens for \"" << doc << "\": " << ss.str();
    }

    // Prepare input for the model as a batch
    vector<vector<string>> batch{tokens};
    vector<torch::jit::IValue> inputs{
        c10mDummyVec, // texts
        c10mDummyVecVec, // multi_texts
        batch, // tokens
        c10mDummyVec // languages
    };

    // Run the model
    auto output =
        mModule.forward(inputs).toGenericListRef().at(0).toGenericDict();

    // Extract and populate results into the response
    for (const auto& elem : output) {
      _return.insert({elem.key().toStringRef(), elem.value().toDouble()});
    }
    VLOG(1) << "Logits for \"" << doc << "\": " << _return;
  }
};

// REST proxy for the predictor Thrift service (not covered in tutorial)
class RestProxyHandler : public Http::Handler {
 private:
  shared_ptr<TTransport> mTransport;
  shared_ptr<PredictorClient> mPredictorClient;
  shared_ptr<Formatter> mFormatter;

  string urlDecode(const string &encoded) {
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

    if (FLAGS_run_tests) {
      mFormatter->runTests();
    }
  }

  void onRequest(const Http::Request& request, Http::ResponseWriter response) {
    if (!mTransport->isOpen()) {
      mTransport->open();
    }

    auto headers = request.headers();

    shared_ptr<Http::Header::ContentType> contentType;
    try {
      contentType = headers.get<Http::Header::ContentType>();
    } catch (runtime_error) {
      response.send(Http::Code::Bad_Request,
                    "Expected HTTP header Content-Type: application/json\n");
      return;
    }

    auto mediaType = contentType->mime();
    if (mediaType != MIME(Application, Json)) {
      response.send(Http::Code::Bad_Request,
                    "Expected HTTP header Content-Type: application/json, found " + mediaType.toString() + "\n");
      return;
    }

    string text;
    try {
      text = mFormatter->formatRequest(request.body());
    } catch (out_of_range e) {
      response.send(Http::Code::Bad_Request,
                    string("Exception: ") + e.what() + "\n");
      return;
    }

    map<string, double> scores;
    mPredictorClient->predict(scores, text);
    response.send(Http::Code::Ok, mFormatter->formatResponse(scores, text));
  }
};

int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  if (argc < 2) {
    cerr << "Usage:" << endl;
    cerr << "./server <DocNN model file>" << endl;
    cerr << "./server <XLM-R model file> <XLM-R sentencepiece vocab file>" << endl;
    return 1;
  }

  string modelFile = argv[1];
  shared_ptr<PredictorHandler> handler;
  if (argc < 3) {
    LOG(INFO) << "Loading monolingual DocNN model from " << modelFile;
    handler = make_shared<PredictorHandler>(modelFile);
  } else {
    LOG(INFO) << "Loading multilingual XLM-R model from " << modelFile;
    string sentencepieceVocab = argv[2];
    handler = make_shared<PredictorHandler>(modelFile, sentencepieceVocab);
  }

  // Handle shutdown events
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = shutdownHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  // Initialize predictor thrift service
  shared_ptr<TProcessor> mSpProcessor(new PredictorProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(FLAGS_port_thrift));
  shared_ptr<TTransportFactory> transportFactory(
      new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  thriftServer = make_unique<TSimpleServer>(
      mSpProcessor, serverTransport, transportFactory, protocolFactory);
  thread thriftThread([&](){ thriftServer->serve(); });
  LOG(INFO) << "Thrift server running at port: " << FLAGS_port_thrift;

  if (FLAGS_rest) {
    // Initialize Thrift client used to foward requests from REST
    shared_ptr<TTransport> socket(new TSocket("127.0.0.1", FLAGS_port_thrift));
    shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    shared_ptr<PredictorClient> predictorClient(new PredictorClient(protocol));

    // Initialize REST proxy
    Address addr(Ipv4::any(), Port(FLAGS_port_rest));
    auto opts = Http::Endpoint::options().threads(1);
    restServer = make_unique<Http::Endpoint>(addr);
    restServer->init(opts);
    restServer->setHandler(
        make_shared<RestProxyHandler>(transport, predictorClient));
    thread restThread([&]() { restServer->serve(); });

    LOG(INFO) << "REST proxy server running at port: " << FLAGS_port_rest;
    restThread.join();
  }

  thriftThread.join();
  return 0;
}
