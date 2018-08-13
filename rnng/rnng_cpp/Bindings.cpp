// Copyright 2004-present Facebook. All Rights Reserved.

#include <torch/csrc/torch.h>
#include "Parser.h"

namespace py = pybind11;
using namespace facebook::assistant::rnng;

PYBIND11_MODULE(rnng_bindings, m) {
  py::class_<Parser, std::shared_ptr<Parser>>(m, "Parser")
      .def(py::init<
           std::vector<float>,            // modelConfig
           std::vector<std::string>,      // actionsVec
           std::vector<std::string>,      // terminalsVec
           std::vector<std::string>>())   // dictfeatsVec
      .def("cuda", &Parser::cuda)
      .def("eval", &Parser::eval)
      .def("forward", &Parser::forward)
      .def("init_word_weights", &Parser::initWordWeights)
      .def("load_state_dict", &Parser::loadStateDict)
      .def("make", &Parser::make)
      .def("parameters", &Parser::parameters)
      .def("predict", &Parser::predict)
      .def("state_dict", &Parser::parameters)
      .def("train", &Parser::train)
      .def("zero_grad", &Parser::zeroGrad);
}
