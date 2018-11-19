// Copyright 2004-present Facebook. All Rights Reserved.

#include "Parser.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include <common/logging/logging.h>

namespace facebook {
namespace assistant {
namespace rnng {
namespace {
torch::Tensor initialState(at::IntList dims, torch::Device device) {
  auto first = torch::zeros(dims, torch::device(device).requires_grad(true));
  auto second = torch::zeros(dims, torch::device(device).requires_grad(true));
  return torch::stack({first, second});
}
} // namespace

const std::string Parser::kStrShift = "SHIFT";
const std::string Parser::kStrReduce = "REDUCE";
const std::string Parser::kStrOpenBracket = "[";
const std::string Parser::kStrCloseBracket = "]";

Parser::StackRNN::StackRNN(
    Parser& parent,
    torch::nn::LSTM lstm,
    const torch::Tensor& state,
    const torch::Tensor& emptyEmbedding)
    : parent_(parent), lstm_(lstm), emptyEmbedding_(emptyEmbedding) {
  // Push in (state, (stateEmbedding, token))
  stack_.push_back(
      std::make_pair(state, std::make_pair(rnnGetOutput_(state), -1)));
  lstm_->flatten_parameters();
}

torch::Tensor Parser::StackRNN::rnnGetOutput_(
    const torch::Tensor& state) const {
  // -1: final layer, 0: hidden_n
  return state[-1][0];
}

torch::Tensor Parser::StackRNN::embedding() {
  if (stack_.size() > 1) {
    return rnnGetOutput_(stack_.back().first);
  }
  return emptyEmbedding_;
}

long Parser::StackRNN::elementFromTop(long index) const {
  return stack_[stack_.size() - index - 1].second.second;
}

void Parser::StackRNN::pop() {
  CAFFE_ENFORCE_GT(stack_.size(), 0);
  stack_.pop_back();
}

void Parser::StackRNN::push(const torch::Tensor& expression, long token) {
  auto expr = expression.unsqueeze(0);
  auto lstmOutput = lstm_->forward(expr, stack_.back().first);
  // Push in (state, (stateEmbedding, token))
  stack_.push_back(std::make_pair(
      lstmOutput.state,
      std::make_pair(rnnGetOutput_(lstmOutput.state), token)));
}

std::pair<torch::Tensor, long> Parser::StackRNN::top() const {
  CAFFE_ENFORCE_GT(stack_.size(), 0);
  return stack_.back().second;
}

void Parser::CompositionalSummationNN::reset() {
  linearSeq_ = register_module(
      "linear_seq",
      torch::nn::Sequential(
          torch::nn::Linear(lstmDim_, lstmDim_),
          torch::nn::Functional(torch::tanh)));
}

torch::Tensor Parser::CompositionalSummationNN::forward(
    std::vector<torch::Tensor> x) {
  auto combined = torch::sum(torch::cat(x), 0, true);
  return linearSeq_->forward(combined);
}

Parser::Parser(
    const std::vector<float>& modelConfig,
    const StringVec& actionsVec,
    const StringVec& terminalsVec,
    const StringVec& dictfeatsVec)
    : modelConfig_(modelConfig),
      actionsVec_(actionsVec),
      terminalsVec_(terminalsVec),
      dictfeatsVec_(dictfeatsVec),
      constraints_(std::make_shared<IntentSlotConstraints>()),
      lstmDim_(long(modelConfig[1])),
      lstmLayers_(long(modelConfig[2])),
      embedDim_(long(modelConfig[3])),
      maxOpenNt_(long(modelConfig[4])),
      dictfeatsEmbedDim_(long(modelConfig[5])),
      dropout_(modelConfig[6]) {
  shiftIdx_ = Preprocessor::findIdx(actionsVec_, kStrShift);
  reduceIdx_ = Preprocessor::findIdx(actionsVec_, kStrReduce);

  addDictFeats_ = dictfeatsEmbedDim_ > 0 && dictfeatsVec_.size() > 0;
  constraints_->init(*this);

  reset();
}

void Parser::reset() {
  wordsLookup_ = register_module(
      "WORDS_LOOKUP", torch::nn::Embedding(terminalsVec_.size(), embedDim_));
  actionsLookup_ = register_module(
      "ACTIONS_LOOKUP", torch::nn::Embedding(actionsVec_.size(), lstmDim_));

  if (addDictFeats_) {
    dictfeatsLookup_ = register_module(
        "DICTFEATS_LOOKUP",
        torch::nn::Embedding(dictfeatsVec_.size(), dictfeatsEmbedDim_));
  }

  // Submodules.
  actionLinear_ = register_module(
      "action_linear",
      torch::nn::Sequential(
          torch::nn::Linear(3 * lstmDim_, lstmDim_),
          torch::nn::Functional(torch::relu),
          torch::nn::Linear(lstmDim_, actionsVec_.size())));
  dropoutLayer_ =
      register_module("dropout_layer", torch::nn::Dropout(dropout_));
  bufferRnn_ = register_module(
      "buff_rnn",
      torch::nn::LSTM(
          torch::nn::LSTMOptions(embedDim_ + dictfeatsEmbedDim_, lstmDim_)
              .layers(lstmLayers_)
              .dropout(dropout_)));
  stackRnn_ = register_module(
      "stack_rnn",
      torch::nn::LSTM(torch::nn::LSTMOptions(lstmDim_, lstmDim_)
                          .layers(lstmLayers_)
                          .dropout(dropout_)));
  actionRnn_ = register_module(
      "action_rnn",
      torch::nn::LSTM(torch::nn::LSTMOptions(lstmDim_, lstmDim_)
                          .layers(lstmLayers_)
                          .dropout(dropout_)));
  compositional_ = register_module(
      "p_compositional", std::make_shared<CompositionalSummationNN>(lstmDim_));

  // Parameters.
  emptyBufferEmb_ = register_parameter(
      "pempty_buffer_emb", torch::randn({1, lstmDim_}, torch::requires_grad()));
  emptyStackEmb_ = register_parameter(
      "empty_stack_emb", torch::randn({1, lstmDim_}, torch::requires_grad()));
  emptyActionEmb_ = register_parameter(
      "empty_action_emb", torch::randn({1, lstmDim_}, torch::requires_grad()));

  compositionalIdx_ = actionsVec_.size() + terminalsVec_.size() + 1;
}

StateMap Parser::getStateDict() {
  auto pairs = named_parameters().pairs();
  return StateMap(pairs.begin(), pairs.end());
}

void Parser::loadStateDict(const StateMap& stateMap) {
  auto params = named_parameters();
  for (const auto& statePair : stateMap) {
    auto* parameter = params.find(statePair.first);
    CAFFE_ENFORCE(parameter);
    torch::NoGradGuard no_grad;
    parameter->copy_(statePair.second);
  }
}

void Parser::initWordWeights(torch::Tensor pretrainedWordWeights) {
  auto params = wordsLookup_->named_parameters();
  params["weight"].set_requires_grad(false);
  params["weight"].copy_(pretrainedWordWeights);
  params["weight"].set_requires_grad(true);
}

PredictorResult Parser::getPredictorResult(
    const StringVec& tokenizedText,
    const StringVec& dictfeatsStrings,
    const FloatVec& dictfeatsWeights,
    const LongVec& dictfeatsLengths) {
  // Initialization
  StringVec actions;
  StringVec tokens;
  FloatVec scores;
  std::string pretty_print;
  LongVec dictfeatsIndices; // Generated from dictfeatsStrings

  // Pre-processing
  if (tokenizedText.size() == 0) {
    return std::make_tuple(actions, tokens, scores, pretty_print);
  }
  auto wordIndices = preprocessor_.getIndices(tokenizedText, terminalsVec_);
  if (addDictFeats_) {
    dictfeatsIndices =
        preprocessor_.getIndices(dictfeatsStrings, dictfeatsVec_, true);
  }

  // Prediction
  std::vector<int> tokenLens;
  for (auto token : tokens) {
    tokenLens.push_back(token.length());
  }
  auto predictionResult = forward(
      torch::tensor(wordIndices).unsqueeze(0),
      torch::tensor(tokenLens).unsqueeze(0),
      std::make_tuple(
          torch::tensor(dictfeatsIndices).unsqueeze(0),
          torch::tensor(
              FloatVec(dictfeatsWeights.begin(), dictfeatsWeights.end()))
              .unsqueeze(0),
          torch::tensor(
              LongVec(dictfeatsLengths.begin(), dictfeatsLengths.end()))));

  auto predictedActions = vecFromVar_<long>(predictionResult[0].squeeze(0));
  auto predictedScores =
      predictionResult[1].squeeze(0); // sequence_length x action_count

  // Post-processing
  predictedScores.exp_();
  auto predictedNormalizedScores =
      std::get<0>(predictedScores.max(1)) / predictedScores.sum(1);
  scores = vecFromVar_<float>(predictedNormalizedScores);

  long tokensIdx = 0;
  for (const auto& action : predictedActions) {
    if (action == reduceIdx_) {
      actions.push_back(kStrReduce);
      tokens.push_back("");
      pretty_print.append(kStrCloseBracket);
    } else if (action == shiftIdx_) {
      actions.push_back(kStrShift);
      tokens.push_back(tokenizedText[tokensIdx]);
      pretty_print.append(tokenizedText[tokensIdx]);
      tokensIdx++;
    } else if (Preprocessor::hasMember(validNtIdxs, action)) {
      std::string nt = actionsVec_[action];
      actions.push_back(constraints_->splitNt(nt).first);
      tokens.push_back(constraints_->splitNt(nt).second);
      pretty_print.append(kStrOpenBracket).append(nt);
    }
    pretty_print.append(" ");
  }
  pretty_print.pop_back(); // Remove trailing whitespace after last ]
  return std::make_tuple(actions, tokens, scores, pretty_print);
}

std::vector<torch::Tensor> Parser::forward(
    torch::Tensor tokens,
    torch::Tensor seqLens,
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dictfeats,
    torch::optional<std::vector<LongVec>> oracleActions) {
  if (!is_training()) {
    torch::manual_seed(0);
  }
  predictedActionsIdx_.clear();

  // Remove the batch dimension and reverse the tensors.
  torch::Tensor tokensReversed = tokens.squeeze(0).flip(0);
  torch::Tensor dictfeatsIndicesReversed =
      std::get<0>(dictfeats).squeeze(0).flip(0);
  torch::Tensor dictfeatsWeightsReversed =
      std::get<1>(dictfeats).squeeze(0).flip(0);
  torch::Tensor dictfeatsLengthsReversed = std::get<2>(dictfeats).flip(0);
  torch::optional<LongVec> oracleActionsReversed;
  if (is_training()) {
    CAFFE_ENFORCE(oracleActions.has_value());
    oracleActionsReversed = oracleActions->at(0);
    std::reverse(oracleActionsReversed->begin(), oracleActionsReversed->end());
  }

  const auto device = dictfeatsIndicesReversed.device();

  std::vector<int64_t> dims = {lstmLayers_, 1, lstmDim_};
  auto bufferInitialState = initialState(dims, device);
  auto stackInitialState = initialState(dims, device);
  auto actionInitialState = initialState(dims, device);

  StackRNN bufferStackRnn(
      *this, bufferRnn_, bufferInitialState, emptyBufferEmb_);
  StackRNN stackStackRnn(*this, stackRnn_, stackInitialState, emptyStackEmb_);
  StackRNN actionStackRnn(
      *this, actionRnn_, actionInitialState, emptyActionEmb_);

  // Prepare Token Embeddings
  auto tokenEmbeddings = wordsLookup_->forward(tokensReversed);
  torch::Tensor dictfeatEmbeddings;
  if (addDictFeats_) {
    if (dictfeatsIndicesReversed.numel() > 0 &&
        dictfeatsWeightsReversed.numel() > 0 &&
        dictfeatsLengthsReversed.numel() > 0) {
      auto dictEmbeddings = dictfeatsLookup_->forward(dictfeatsIndicesReversed);

      // Reshape weighted dict feats such that first axis is num_tokens.
      int maxTokens = tokensReversed.size(0);
      auto weightedDictEmbeddings =
          dictEmbeddings.mul(dictfeatsWeightsReversed.unsqueeze(1))
              .view(torch::IntList{maxTokens, -1, dictfeatsEmbedDim_});

      // Add extra dim for embedding and cast to float
      dictfeatsLengthsReversed = dictfeatsLengthsReversed.unsqueeze(1).to(
          weightedDictEmbeddings.scalar_type());

      // Average pooling of dict feats.
      dictfeatEmbeddings =
          weightedDictEmbeddings.sum(1).div(dictfeatsLengthsReversed);

    } else {
      dictfeatEmbeddings = torch::zeros(
          {tokenEmbeddings.size(0), dictfeatsEmbedDim_},
          torch::requires_grad());
    }

    // Final token embedding is concatenation of word and dict feat
    // embeddings.
    tokenEmbeddings = torch::cat({tokenEmbeddings, dictfeatEmbeddings}, 1);
  }

  // Initialize Buffer
  for (size_t i = 0; i < tokenEmbeddings.size(0); i++) {
    bufferStackRnn.push(
        tokenEmbeddings[i].unsqueeze(0),
        vecFromVar_<long>(tokensReversed[i])[0]);
  }

  std::vector<torch::Tensor> action_scores;
  bool ignoreLossForRoot = false;
  long numOpenNt = 0;
  isOpenNt_.clear();

  while (!(stackStackRnn.size() == 1 && bufferStackRnn.size() == 0)) {
    auto validActions =
        getValidActions_(stackStackRnn, bufferStackRnn, numOpenNt).to(device);

    torch::Tensor bufferSummary = bufferStackRnn.embedding();
    torch::Tensor stackSummary = stackStackRnn.embedding();
    torch::Tensor actionSummary = actionStackRnn.embedding();

    if (dropout_ > 0) {
      bufferSummary = dropoutLayer_->forward(bufferSummary);
      stackSummary = dropoutLayer_->forward(stackSummary);
      actionSummary = dropoutLayer_->forward(actionSummary);
    }

    actionP_ = torch::cat({bufferSummary, stackSummary, actionSummary}, 1);
    actionP_ = actionLinear_->forward(actionP_);

    auto logProbs = log_softmax(actionP_, 1).squeeze(0);
    auto logProbsLegal = logProbs.index_select(0, validActions);
    auto predictedActionIdxsLegal = std::get<1>(max(logProbsLegal, 0));
    auto predictedActionIdxVar =
        validActions.index_select(0, predictedActionIdxsLegal);
    long predictedActionIdx = vecFromVar_<long>(predictedActionIdxVar)[0];
    predictedActionsIdx_.push_back(predictedActionIdx);

    long targetActionIdx = predictedActionIdx;
    if (is_training()) {
      CAFFE_ENFORCE(oracleActionsReversed.has_value());
      CAFFE_ENFORCE_GT(oracleActionsReversed->size(), 0);
      targetActionIdx = oracleActionsReversed->back();
      oracleActionsReversed->pop_back();
    }
    if (!ignoreLossForRoot) {
      action_scores.push_back(actionP_);
    }

    auto actionEmbedding = actionsLookup_->forward(predictedActionIdxVar);
    actionStackRnn.push(actionEmbedding, predictedActionIdx);

    if (targetActionIdx == shiftIdx_) {
      // To SHIFT,
      // 1. Pop T from buffer
      // 2. Push T into stack

      isOpenNt_.push_back(false);
      auto tokenEmbedding = bufferStackRnn.top().first;
      long token = bufferStackRnn.top().second;
      bufferStackRnn.pop();
      stackStackRnn.push(tokenEmbedding, token);

    } else if (targetActionIdx == reduceIdx_) {
      // To REDUCE
      // 1. Pop Ts from stack until hit NT
      // 2. Pop the open NT from stack and close it
      // 3. Compute compositionalRep and push into stack

      --numOpenNt;
      poppedRep_.clear();
      ntTree_.clear();

      while (!isOpenNt_.back()) {
        CAFFE_ENFORCE_GT(stackStackRnn.size(), 0);
        isOpenNt_.pop_back();
        auto topOfStack = stackStackRnn.top();
        poppedRep_.push_back(topOfStack.first);
        ntTree_.push_back(topOfStack.second);
        stackStackRnn.pop();
      }

      auto topOfStack = stackStackRnn.top();
      poppedRep_.push_back(topOfStack.first);
      ntTree_.push_back(topOfStack.second);
      stackStackRnn.pop();

      isOpenNt_.pop_back();
      isOpenNt_.push_back(false);

      auto compositionalRep = compositional_->forward(poppedRep_);

      stackStackRnn.push(compositionalRep, compositionalIdx_++);

    } else if (Preprocessor::hasMember(validNtIdxs, targetActionIdx)) {
      // To insert action, push actionEmbedding into stack

      if (predictedActionsIdx_.size() == 1 &&
          Preprocessor::hasMember(ignoreSubNtRoots, targetActionIdx)) {
        ignoreLossForRoot = true;
      }

      isOpenNt_.push_back(true);
      ++numOpenNt;
      stackStackRnn.push(actionEmbedding, predictedActionIdx);

    } else {
      LOG(FATAL) << "Predicted an invalid action";
    }
  }

  CAFFE_ENFORCE_EQ(stackStackRnn.size(), 1);
  CAFFE_ENFORCE_EQ(bufferStackRnn.size(), 0);

  // Add batch dimension before returning.
  return {torch::tensor(predictedActionsIdx_).unsqueeze(0),
          torch::cat(action_scores).unsqueeze(0)};
}

torch::Tensor Parser::getValidActions_(
    const StackRNN& stack,
    const StackRNN& buffer,
    long numOpenNt) {
  validActions_.clear();

  // Can SHIFT if
  // 1. Buffer is non-empty, and
  // 2. At least one open NT on stack
  if (buffer.size() > 0 && numOpenNt >= 1) {
    validActions_.push_back(shiftIdx_);
  }

  // Can REDUCE if
  // 1. Top of stack is not an NT, and
  // 2. Two open NTs on stack, or buffer is empty
  if ((isOpenNt_.size() > 0 && !isOpenNt_.back()) &&
      (numOpenNt >= 2 || buffer.size() == 0)) {
    CAFFE_ENFORCE_GT(stack.size(), 0);
    validActions_.push_back(reduceIdx_);
  }

  // Can perform ACTION if constraints say so
  if (buffer.size() > 0 && numOpenNt < maxOpenNt_) {
    long lastOpenNt = -1;

    std::reverse(isOpenNt_.begin(), isOpenNt_.end());
    auto location = std::find(isOpenNt_.begin(), isOpenNt_.end(), true);
    if (location != isOpenNt_.end()) {
      lastOpenNt = stack.elementFromTop(location - isOpenNt_.begin());
    }
    std::reverse(isOpenNt_.begin(), isOpenNt_.end());

    constraints_->populateActions(*this, lastOpenNt);

  } else if (numOpenNt >= maxOpenNt_) {
    VLOG(1) << "Not predicting NT, numOpenNts is " << numOpenNt;
  }

  CAFFE_ENFORCE_GT(validActions_.size(), 0);
  return torch::tensor(validActions_);
}

const std::string IntentSlotConstraints::kStrIntent = "IN:";
const std::string IntentSlotConstraints::kStrSlot = "SL:";

} // namespace rnng
} // namespace assistant
} // namespace facebook
