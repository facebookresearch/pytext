// Copyright 2004-present Facebook. All Rights Reserved.

#include "Parser.h"
#include "common/logging/logging.h"

namespace facebook {
namespace assistant {
namespace rnng {

const std::string Parser::kStrShift = "SHIFT";
const std::string Parser::kStrReduce = "REDUCE";
const std::string Parser::kStrOpenBracket = "[";
const std::string Parser::kStrCloseBracket = "]";

Parser::StackRNN::StackRNN(
    Parser& parent,
    autograd::Container lstm,
    const autograd::Variable& state,
    const autograd::Variable& emptyEmbedding)
    : parent_(parent), lstm_(lstm), emptyEmbedding_(emptyEmbedding) {
  // Push in (state, (stateEmbedding, token))
  stack_.push_back(
      std::make_pair(state, std::make_pair(rnnGetOutput_(state), -1)));
}

autograd::Variable Parser::StackRNN::rnnGetOutput_(
    const autograd::Variable& state) const {
  // -1: final layer, 0: hidden_n
  if (parent_.cuda_ && (parent_.dropout_ == 0)) {
    // CUDA has layers as second dimension, not first
    return state[0][-1];
  }
  return state[-1][0];
}

autograd::Variable Parser::StackRNN::embedding() {
  if (stack_.size() > 1) {
    return rnnGetOutput_(stack_.back().first);
  }
  return emptyEmbedding_;
}

long Parser::StackRNN::elementFromTop(long index) const {
  return stack_[stack_.size() - index - 1].second.second;
}

void Parser::StackRNN::pop() {
  CHECK_GT(stack_.size(), 0);
  stack_.pop_back();
}

void Parser::StackRNN::push(const autograd::Variable& expression, long token) {
  auto expr = expression.unsqueeze(0);
  auto lstmOutput = lstm_->forward({expr, stack_.back().first});
  // Push in (state, (stateEmbedding, token))
  CHECK_GT(lstmOutput.size(), 0);
  stack_.push_back(std::make_pair(
      lstmOutput[1], std::make_pair(rnnGetOutput_(lstmOutput[1]), token)));
}

std::pair<autograd::Variable, long> Parser::StackRNN::top() const {
  CHECK_GT(stack_.size(), 0);
  return stack_.back().second;
}

Parser::Parser(
    const std::vector<float>& modelConfig,
    const StringVec& actionsVec,
    const StringVec& terminalsVec,
    const StringVec& dictfeatsVec)
    : lstmDim_(long(modelConfig[1])),
      lstmLayers_(long(modelConfig[2])),
      embedDim_(long(modelConfig[3])),
      maxOpenNt_(long(modelConfig[4])),
      dictfeatsEmbedDim_(long(modelConfig[5])),
      dropout_(modelConfig[6]),
      actionsVec_(actionsVec),
      terminalsVec_(terminalsVec),
      dictfeatsVec_(dictfeatsVec) {
  shiftIdx_ = Preprocessor::findIdx(actionsVec_, kStrShift);
  reduceIdx_ = Preprocessor::findIdx(actionsVec_, kStrReduce);
  preprocessor_ = Preprocessor();

  addDictFeats_ = dictfeatsEmbedDim_ > 0 && dictfeatsVec_.size() > 0;

  constraints_ = std::make_shared<IntentSlotConstraints>();
  constraints_->init(*this);
}

void Parser::initialize_containers() {
  wordsLookup_ =
      add(autograd::Embedding(terminalsVec_.size(), embedDim_).make(),
          "WORDS_LOOKUP");
  actionsLookup_ =
      add(autograd::Embedding(actionsVec_.size(), lstmDim_).make(),
          "ACTIONS_LOOKUP");

  if (addDictFeats_) {
    dictfeatsLookup_ = add(
        autograd::Embedding(dictfeatsVec_.size(), dictfeatsEmbedDim_).make(),
        "DICTFEATS_LOOKUP");
  }

  actionLinear0_ =
      add(autograd::Linear(3 * lstmDim_, lstmDim_).make(), "action_linear/0");
  actionLinear1_ = add(
      autograd::Linear(lstmDim_, actionsVec_.size()).make(), "action_linear/2");
  dropoutLayer_ = add(autograd::Dropout(dropout_).make(), "dropout_layer");
  bufferRnn_ =
      add(autograd::LSTM(embedDim_ + dictfeatsEmbedDim_, lstmDim_)
              .nlayers(lstmLayers_)
              .dropout(dropout_)
              .make(),
          "buff_rnn");
  stackRnn_ =
      add(autograd::LSTM(lstmDim_, lstmDim_)
              .nlayers(lstmLayers_)
              .dropout(dropout_)
              .make(),
          "stack_rnn");
  actionRnn_ =
      add(autograd::LSTM(lstmDim_, lstmDim_)
              .nlayers(lstmLayers_)
              .dropout(dropout_)
              .make(),
          "action_rnn");
  compositional_ =
      add(autograd::Linear(lstmDim_, lstmDim_).make(),
          "p_compositional/linear_seq/0");
}

void Parser::initialize_parameters() {
  emptyBufferEmb_ =
      add(autograd::Var(DefaultTensor(at::kFloat).randn({1, lstmDim_})),
          "pempty_buffer_emb");
  emptyStackEmb_ =
      add(autograd::Var(DefaultTensor(at::kFloat).randn({1, lstmDim_})),
          "empty_stack_emb");
  emptyActionEmb_ =
      add(autograd::Var(DefaultTensor(at::kFloat).randn({1, lstmDim_})),
          "empty_action_emb");

  compositionalIdx_ = actionsVec_.size() + terminalsVec_.size() + 1;

  // Initialize CUDA
  actionP_ = autograd::Var(DefaultTensor(at::kFloat).zeros({lstmDim_}));
  actionP_ = actionLinear1_->forward({actionP_})[0];
}

autograd::Variable Parser::xavierInitialState_(const at::IntList& dims) {
  double std = std::sqrt(2.0 / fanInOut_(dims));
  auto tensor0 =
      (autograd::Var(DefaultTensor(at::kFloat).tensor(dims).normal_(0, std)));
  auto tensor1 =
      (autograd::Var(DefaultTensor(at::kFloat).tensor(dims).normal_(0, std)));
  return at::stack({tensor0, tensor1}, cuda_ && (dropout_ == 0) ? 0 : 1);
}

void Parser::loadStateDict(const StateMap& stateMap, bool transformNames) {
  auto params = parameters();

  // Ugly temporary hacky remapping from PyTorch naming to autogradpp naming.
  // TODO: (mrinalmohit) T28207528
  // Move from autogradpp to PyTorch C++ API to fix this
  RE2 rgxDot("\\.");
  std::string fmtDot("/");
  RE2 rgxRnn("(\\w+)_([ih])([ih])_l(\\d)");
  std::string type, in, out;
  int layer;

  for (auto const& statePair : stateMap) {
    std::string label = statePair.first;

    if (transformNames) {
      RE2::GlobalReplace(&label, rgxDot, fmtDot);
      if (RE2::PartialMatch(label, rgxRnn, &type, &in, &out, &layer)) {
        std::string fmtRnn =
            in + "2" + out + "_" + std::to_string(layer) + "/" + type;
        RE2::Replace(&label, rgxRnn, fmtRnn);
      }
    }

    auto it = params.find(label);
    if (it != params.end()) {
      params[label].set_requires_grad(false);
      params[label].copy_(statePair.second);
      params[label].set_requires_grad(true);
    } else {
      LOG(WARNING) << label << " not found in parameter list";
    }
  }
}

void Parser::initWordWeights(at::Tensor pretrainedWordWeights) {
  auto params = wordsLookup_->parameters();
  params["weight"].set_requires_grad(false);
  params["weight"].copy_(pretrainedWordWeights);
  params["weight"].set_requires_grad(true);
}

void Parser::zeroGrad() {
  for (auto p : parameters()) {
    auto& grad = p.second.grad();
    if (grad.defined()) {
      grad = grad.detach();
      grad.zero_();
    }
  }
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

  std::reverse(wordIndices.begin(), wordIndices.end());
  std::reverse(dictfeatsIndices.begin(), dictfeatsIndices.end());

  // Prediction
  auto predictionResult =
      forward({varFromVec_<long>(wordIndices, at::kLong),
               varFromVec_<long>(dictfeatsIndices, at::kLong),
               varFromVec_<float>(
                   FloatVec(dictfeatsWeights.rbegin(), dictfeatsWeights.rend()),
                   at::kFloat), // Can't reverse a const vector in-place
               varFromVec_<long>(
                   LongVec(dictfeatsLengths.rbegin(), dictfeatsLengths.rend()),
                   at::kLong)}); // Can't reverse a const vector in-place

  auto predictedActions = vecFromVar_<long>(predictionResult[0]);
  auto predictedScores = predictionResult[1]; // sequence_length x action_count

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

std::vector<autograd::Variable> Parser::forward(
    std::vector<autograd::Variable> x) {
  CHECK_GT(x.size(), 3);

  auto tokensReversed = x[0];
  auto dictfeatsIndicesReversed = x[1];
  auto dictfeatsWeightsReversed = x[2];
  auto dictfeatsLengthsReversed = x[3];

  LongVec oracleActionsReversed;
  if (train_) {
    CHECK_GT(x.size(), 4);
    oracleActionsReversed = vecFromVar_<long>(x[4]);
  }

  predictedActionsIdx_.clear();

  autograd::Variable bufferInitialState =
      xavierInitialState_({lstmLayers_, 1, lstmDim_});
  autograd::Variable stackInitialState =
      xavierInitialState_({lstmLayers_, 1, lstmDim_});
  autograd::Variable actionInitialState =
      xavierInitialState_({lstmLayers_, 1, lstmDim_});

  StackRNN bufferStackRnn =
      StackRNN(*this, bufferRnn_, bufferInitialState, emptyBufferEmb_);
  StackRNN stackStackRnn =
      StackRNN(*this, stackRnn_, stackInitialState, emptyStackEmb_);
  StackRNN actionStackRnn =
      StackRNN(*this, actionRnn_, actionInitialState, emptyActionEmb_);

  // Prepare Token Embeddings
  auto tokenEmbeddings = wordsLookup_->forward({tokensReversed})[0];
  autograd::Variable dictfeatEmbeddings;
  if (addDictFeats_) {
    if (dictfeatsIndicesReversed.numel() > 0 &&
        dictfeatsWeightsReversed.numel() > 0 &&
        dictfeatsLengthsReversed.numel() > 0) {
      auto dictEmbeddings =
          dictfeatsLookup_->forward({dictfeatsIndicesReversed})[0];

      // Reshape weighted dict feats such that first axis is num_tokens.
      int maxTokens = tokensReversed.sizes()[0];
      auto weightedDictEmbeddings =
          dictEmbeddings.mul(dictfeatsWeightsReversed.unsqueeze(1))
              .view(at::IntList{maxTokens, -1, dictfeatsEmbedDim_});

      // Add extra dim for embedding and cast to float
      dictfeatsLengthsReversed = dictfeatsLengthsReversed.unsqueeze(1).toType(
          weightedDictEmbeddings.type());

      // Average pooling of dict feats.
      dictfeatEmbeddings =
          weightedDictEmbeddings.sum(1).div(dictfeatsLengthsReversed);

    } else {
      dictfeatEmbeddings = autograd::Var(
          DefaultTensor(at::kFloat)
              .zeros({tokenEmbeddings.sizes()[0], dictfeatsEmbedDim_}));
    }

    // Final token embedding is concatenation of word and dict feat embeddings.
    tokenEmbeddings = at::cat({tokenEmbeddings, dictfeatEmbeddings}, 1);
  }

  // Initialize Buffer
  for (size_t i = 0; i < tokenEmbeddings.sizes()[0]; i++) {
    bufferStackRnn.push(
        tokenEmbeddings[i].unsqueeze(0),
        vecFromVar_<long>(tokensReversed[i])[0]);
  }

  std::vector<at::Tensor> action_scores;
  bool ignoreLossForRoot = false;
  long numOpenNt = 0;
  isOpenNt_.clear();

  while (!(stackStackRnn.size() == 1 && bufferStackRnn.size() == 0)) {
    auto validActions =
        getValidActions_(stackStackRnn, bufferStackRnn, numOpenNt);

    autograd::Variable bufferSummary = bufferStackRnn.embedding();
    autograd::Variable stackSummary = stackStackRnn.embedding();
    autograd::Variable actionSummary = actionStackRnn.embedding();

    if (dropout_ > 0) {
      auto dropoutResult =
          dropoutLayer_->forward({bufferSummary, stackSummary, actionSummary});
      CHECK_GT(dropoutResult.size(), 2);
      bufferSummary = dropoutResult[0];
      stackSummary = dropoutResult[1];
      actionSummary = dropoutResult[2];
    }

    actionP_ = at::cat({bufferSummary, stackSummary, actionSummary}, 1);
    actionP_ = actionLinear0_->forward({actionP_})[0];
    actionP_ = actionP_.clamp_min(0); // ReLU
    // TODO: (mrinalmohit) T28867071 Replace ReLU with a Container wrapper
    actionP_ = actionLinear1_->forward({actionP_})[0];

    auto logProbs = log_softmax(actionP_, 1).squeeze(0);
    auto logProbsLegal = logProbs.index_select(0, validActions);
    auto predictedActionIdxsLegal = std::get<1>(max(logProbsLegal, 0));
    auto predictedActionIdxVar =
        validActions.index_select(0, predictedActionIdxsLegal);
    long predictedActionIdx = vecFromVar_<long>(predictedActionIdxVar)[0];
    predictedActionsIdx_.push_back(predictedActionIdx);

    long targetActionIdx = predictedActionIdx;
    if (train_) {
      CHECK_GT(oracleActionsReversed.size(), 0);
      targetActionIdx = oracleActionsReversed.back();
      oracleActionsReversed.pop_back();
    }
    if (!ignoreLossForRoot) {
      action_scores.push_back(actionP_);
    }

    auto actionEmbedding = actionsLookup_->forward({predictedActionIdxVar})[0];
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
      // 3. Combine popped Ts into a combinedRep, compute compositionalRep
      // 4. Push compositionalRep into stack

      --numOpenNt;
      poppedRep_.clear();
      ntTree_.clear();

      while (!isOpenNt_.back()) {
        CHECK_GT(stackStackRnn.size(), 0);
        isOpenNt_.pop_back();
        auto topOfStack = stackStackRnn.top();
        poppedRep_.push_back(topOfStack.first.data());
        ntTree_.push_back(topOfStack.second);
        stackStackRnn.pop();
      }

      auto topOfStack = stackStackRnn.top();
      poppedRep_.push_back(topOfStack.first.data());
      ntTree_.push_back(topOfStack.second);
      stackStackRnn.pop();

      isOpenNt_.pop_back();
      isOpenNt_.push_back(false);

      auto combinedRep = sum(at::cat(poppedRep_), 0, true);
      auto compositionalRep =
          compositional_->forward({autograd::Var(combinedRep)})[0];
      compositionalRep = compositionalRep.tanh_();

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

  CHECK_EQ(stackStackRnn.size(), 1);
  CHECK_EQ(bufferStackRnn.size(), 0);
  return {varFromVec_<long>(predictedActionsIdx_, at::kLong),
          at::cat(action_scores)};
}

autograd::Variable Parser::getValidActions_(
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
    CHECK_GT(stack.size(), 0);
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

  CHECK_GT(validActions_.size(), 0);
  return varFromVec_<long>(validActions_, at::kLong);
}

long Parser::fanInOut_(const at::IntList& dims) {
  if (dims.size() == 2) {
    return dims[0] + dims[1];
  }

  if (dims.size() > 2) {
    long receptiveFieldSize = 1;
    for (size_t dim_idx = 2; dim_idx < dims.size(); ++dim_idx) {
      receptiveFieldSize *= dims[dim_idx];
    }
    return receptiveFieldSize * (dims[0] + dims[1]);
  }

  return -1;
}

const std::string IntentSlotConstraints::kStrIntent = "IN:";
const std::string IntentSlotConstraints::kStrSlot = "SL:";

} // namespace rnng
} // namespace assistant
} // namespace facebook
