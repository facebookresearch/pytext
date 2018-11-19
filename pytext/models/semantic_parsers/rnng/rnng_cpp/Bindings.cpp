// Copyright 2004-present Facebook. All Rights Reserved.

#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include <common/logging/logging.h>

#include <torch/python.h>
#include <torch/torch.h>

#include "Parser.h"

namespace py = pybind11;
using namespace facebook::assistant::rnng;

PYBIND11_MODULE(rnng_bindings, m) {
  torch::python::bind_module<Parser>(m, "Parser")
      .def(py::init<
           std::vector<float>, // modelConfig
           std::vector<std::string>, // actionsVec
           std::vector<std::string>, // terminalsVec
           std::vector<std::string>>()) // dictfeatsVec
      .def("forward", &Parser::forward)
      .def("init_word_weights", &Parser::initWordWeights)
      .def("load_state_dict", &Parser::loadStateDict)
      .def("predict", &Parser::predict)
      .def(
          "state_dict",
          [](Parser& parser) {
            auto pairs = parser.named_parameters().pairs();
            return std::unordered_map<std::string, torch::Tensor>(
                pairs.begin(), pairs.end());
          })
      .def(py::pickle(
          [](const std::shared_ptr<Parser> parser) { // __getstate__
            LOG(INFO) << "Deserializing model";
            // State dict: paramsMap
            auto pairs = parser->named_parameters().pairs();
            auto paramsMap = std::unordered_map<std::string, torch::Tensor>(
                pairs.begin(), pairs.end());

            return py::make_tuple(
                parser->modelConfig_,
                parser->actionsVec_,
                parser->terminalsVec_,
                parser->dictfeatsVec_,
                paramsMap);
          },
          [](py::tuple tup) { // __setstate__
            LOG(INFO) << "Serializing model";
            std::shared_ptr<Parser> parser = std::make_shared<Parser>(
                tup[0].cast<std::vector<float>>(),
                tup[1].cast<std::vector<std::string>>(),
                tup[2].cast<std::vector<std::string>>(),
                tup[3].cast<std::vector<std::string>>());
            parser->loadStateDict(
                tup[4].cast<facebook::assistant::rnng::StateMap>());
            return parser;
          }));
}
