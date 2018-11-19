// Copyright 2004-present Facebook. All Rights Reserved.

#include "pytext/models/semantic_parsers/rnng/rnng_cpp/Parser.h"
#include "caffe2/caffe2/contrib/aten/aten_op.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace fb {
namespace {

enum RNNG_INPUTS {
  RNNG_TOKENS_VALS_STR,
  RNNG_DICT_VALS_STR,
  RNNG_DICT_WEIGHTS,
  RNNG_DICT_LENS,
  RNNG_INPUTS_COUNT
};

enum RNNG_OUTPUTS {
  RNNG_ACTIONS,
  RNNG_TOKENS,
  RNNG_SCORES,
  RNNG_PRETTY_PRINT,
  RNNG_OUTPUTS_COUNT
};

template <class Context>
class RNNGParserOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RNNGParserOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    auto modelConfig = OperatorBase::GetRepeatedArgument<float>("model_config");
    auto weightNames =
        OperatorBase::GetRepeatedArgument<std::string>("weight_names");
    auto actionsVec =
        OperatorBase::GetRepeatedArgument<std::string>("actions_vec");
    auto terminalsVec =
        OperatorBase::GetRepeatedArgument<std::string>("terminals_vec");
    auto dictfeatsVec =
        OperatorBase::GetRepeatedArgument<std::string>("dictfeats_vec");

    facebook::assistant::rnng::StateMap stateMap;
    for (size_t i = 0; i < weightNames.size(); ++i) {
      stateMap.insert(
          {weightNames[i],
           torch::autograd::make_variable(
               loadInput(i + RNNG_INPUTS_COUNT), /*requires_grad=*/true)});
    } // Ignore RNNG_INPUTS

    parser_ = std::make_shared<facebook::assistant::rnng::Parser>(
        modelConfig, actionsVec, terminalsVec, dictfeatsVec);
    parser_->loadStateDict(stateMap);
    parser_->eval();
  }

  bool RunOnDevice() override {
    // Get Input Handles
    auto input_tokens_vals_str =
        GetRepeatedInput<std::string>(RNNG_TOKENS_VALS_STR);
    auto input_dict_vals_str =
        GetRepeatedInput<std::string>(RNNG_DICT_VALS_STR);
    auto input_dict_weights = GetRepeatedInput<float>(RNNG_DICT_WEIGHTS);
    auto input_dict_lens = GetRepeatedInput<long>(RNNG_DICT_LENS);

    // Run model to get predicton
    auto prediction = parser_->getPredictorResult(
        input_tokens_vals_str,
        input_dict_vals_str,
        input_dict_weights,
        input_dict_lens);
    auto prediction_actions = std::get<RNNG_ACTIONS>(prediction);
    auto prediction_tokens = std::get<RNNG_TOKENS>(prediction);
    auto prediction_scores = std::get<RNNG_SCORES>(prediction);

    // Get Output Handles
    auto* output_actions = Output(RNNG_ACTIONS);
    auto* output_tokens = Output(RNNG_TOKENS);
    auto* output_scores = Output(RNNG_SCORES);
    auto* output_pretty_print = Output(RNNG_PRETTY_PRINT);

    // Assign prediction to outputs
    output_actions->Resize(prediction_actions.size());
    std::copy(
        prediction_actions.begin(),
        prediction_actions.end(),
        output_actions->template mutable_data<std::string>());

    output_tokens->Resize(prediction_tokens.size());
    std::copy(
        prediction_tokens.begin(),
        prediction_tokens.end(),
        output_tokens->template mutable_data<std::string>());

    output_scores->Resize(prediction_scores.size());
    std::copy(
        prediction_scores.begin(),
        prediction_scores.end(),
        output_scores->template mutable_data<float>());

    output_pretty_print->Resize(1);
    output_pretty_print->template mutable_data<std::string>()[0] =
        std::get<RNNG_PRETTY_PRINT>(prediction);

    return true;
  }

 protected:
  std::shared_ptr<facebook::assistant::rnng::Parser> parser_;

  // Copy-Pasted from private section of contrib/aten/aten_op_template.h:ATenOp
  // TODO: (ezyang) Remove wrapping entirely when at::Tensor and caffe2::Tensor unify

  at::Tensor tensorWrapping(const Tensor& ten_) {
    auto& ten = const_cast<Tensor&>(ten_);
    return at::from_blob(
        ten.raw_mutable_data(),
        ten.sizes(),
        at::device(at::kCPU).dtype(at::typeMetaToScalarType(ten.dtype())));
  }
  at::Tensor loadInput(size_t i) {
    return tensorWrapping(Input(i));
  }

  // End of Copy-Paste

  template <typename T>
  T GetSingleInput(size_t idx) {
    auto& input = Input(idx);
    T input_data = input.template data<T>()[0];
    return input_data;
  }

  template <typename T>
  std::vector<T> GetRepeatedInput(size_t idx, size_t size = 0) {
    auto& input = Input(idx);
    auto* input_data_ptr = input.template data<T>();
    if (size == 0) {
      size = input.numel();
    }
    std::vector<T> input_data(input_data_ptr, input_data_ptr + size);
    return input_data;
  }
};

REGISTER_CPU_OPERATOR(RNNGParser, RNNGParserOp<CPUContext>)

OPERATOR_SCHEMA(RNNGParser)
    .NumInputs(RNNG_INPUTS_COUNT, INT_MAX)
    .NumOutputs(RNNG_OUTPUTS_COUNT)
    .Output(RNNG_ACTIONS, "actions", "Predicted Actions")
    .Output(RNNG_TOKENS, "tokens", "Tokens corresponding to actions")
    .Output(RNNG_SCORES, "scores", "Confidence scores corresponding to actions")
    .Output(RNNG_PRETTY_PRINT, "pretty_print", "Prediction in seqlogical form")
    .Arg(
        "model_config",
        "version, terminal_count, lstm_dim, lstm_layers, \
         embed_dim, max_open_NT, dictfeats_embed_dim, dropout")
    .Arg("weight_names", "vector of names of weight tensors")
    .Arg("actions_vec", "vector of names of actions, sorted by idx")
    .Arg("terminals_vec", "vector of names of terminals, sorted by idx")
    .Arg("dictfeats_vec", "vector of names of dictfeats, sorted by idx");

NO_GRADIENT(RNNGParser);
} // namespace
} // namespace fb
} // namespace caffe2
