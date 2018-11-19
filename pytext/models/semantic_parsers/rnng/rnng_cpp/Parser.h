// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "Preprocessor.h"

namespace facebook {
namespace assistant {
namespace rnng {

using StateMap = std::unordered_map<std::string, torch::Tensor>;
using StateEmbedding = std::pair<torch::Tensor, long>;
using StackElement = std::pair<torch::Tensor, StateEmbedding>;

using LongVec = std::vector<long>;
using FloatVec = std::vector<float>;
using StringVec = std::vector<std::string>;

// PredictorResult = tuple<actions, tokens, scores, pretty_print>
using PredictorResult = std::tuple<StringVec, StringVec, FloatVec, std::string>;

class Parser : public torch::nn::Cloneable<Parser> {
 public:
  Parser(
      const FloatVec& modelConfig,
      // {version, lstm_dim, lstm_layers, embed_dim,
      // max_open_NT, dictfeats_embed_dim, dropout}
      const StringVec& actionsVec,
      const StringVec& terminalsVec,
      const StringVec& dictfeatsVec = std::vector<std::string>());

  void reset() override;

  StateMap getStateDict();
  void loadStateDict(const StateMap& stateMap);
  void initWordWeights(torch::Tensor pretrainedWordWeights);

  std::string predict(
      const StringVec& tokenizedText,
      const StringVec& dictfeatsStrings,
      const FloatVec& dictfeatsWeights,
      const LongVec& dictfeatsLengths) {
    return std::get<3>(getPredictorResult(
        tokenizedText, dictfeatsStrings, dictfeatsWeights, dictfeatsLengths));
  }

  PredictorResult getPredictorResult(
      const StringVec& tokenizedText,
      const StringVec& dictfeatsStrings,
      const FloatVec& dictfeatsWeights,
      const LongVec& dictfeatsLengths);

  std::vector<torch::Tensor> forward(
      torch::Tensor tokens,
      torch::Tensor seqLens,
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dictfeats,
      torch::optional<std::vector<LongVec>> oracleActions = c10::nullopt);

  static const std::string kStrShift;
  static const std::string kStrReduce;
  static const std::string kStrOpenBracket;
  static const std::string kStrCloseBracket;

  std::vector<float> modelConfig_;
  StringVec actionsVec_;
  StringVec terminalsVec_;
  StringVec dictfeatsVec_;

  // Subclass this to define task-specific constraints
  // e.g. IntentSlotConstraints
  class Constraints {
   public:
    virtual void init(Parser& parent) = 0;
    virtual void populateActions(Parser& parent, long lastOpenNt) = 0;
    virtual std::pair<std::string, std::string> splitNt(std::string nt) = 0;
    virtual ~Constraints() {}
  };

  // Exposed as public fields to be modifiable by Constraints objects
  LongVec ignoreSubNtRoots;
  LongVec validNtIdxs;
  LongVec validActions_;
  const StringVec& getActionsVec() const {
    return actionsVec_;
  }

 protected:
  class StackRNN {
   public:
    StackRNN(
        Parser& parent,
        torch::nn::LSTM lstm,
        const torch::Tensor& state,
        const torch::Tensor& emptyEmbedding);

    torch::Tensor embedding();
    long elementFromTop(long index) const;
    void pop();
    void push(const torch::Tensor& expression, long token);
    std::pair<torch::Tensor, long> top() const;

    inline size_t size() const {
      return stack_.size() - 1; // Ignoring empty embedding
    }

   private:
    Parser& parent_;
    torch::nn::LSTM lstm_;
    torch::Tensor emptyEmbedding_;
    std::vector<StackElement> stack_;
    torch::Tensor rnnGetOutput_(const torch::Tensor& state) const;
  };

  class CompositionalSummationNN
      : public torch::nn::Cloneable<CompositionalSummationNN> {
   public:
    CompositionalSummationNN(long lstmDim) : lstmDim_(lstmDim) {
      reset();
    }
    void reset() override;
    torch::Tensor forward(std::vector<torch::Tensor> x);

   protected:
    long lstmDim_;
    torch::nn::Sequential linearSeq_{nullptr};
  };

  Preprocessor preprocessor_;
  std::shared_ptr<CompositionalSummationNN> compositional_;
  std::shared_ptr<Constraints> constraints_;

  torch::nn::Embedding wordsLookup_{nullptr};
  torch::nn::Embedding actionsLookup_{nullptr};
  torch::nn::Embedding dictfeatsLookup_{nullptr};
  torch::nn::Sequential actionLinear_{nullptr};
  torch::nn::Dropout dropoutLayer_{nullptr};
  torch::nn::LSTM bufferRnn_{nullptr};
  torch::nn::LSTM stackRnn_{nullptr};
  torch::nn::LSTM actionRnn_{nullptr};
  torch::Tensor emptyBufferEmb_;
  torch::Tensor emptyStackEmb_;
  torch::Tensor emptyActionEmb_;
  torch::Tensor actionP_;

  bool addDictFeats_ = false;
  long lstmDim_;
  long lstmLayers_;
  long embedDim_;
  long maxOpenNt_;
  long dictfeatsEmbedDim_;
  float dropout_;

  long shiftIdx_;
  long reduceIdx_;
  long compositionalIdx_;

  LongVec predictedActionsIdx_;
  std::vector<bool> isOpenNt_;
  std::vector<torch::Tensor> poppedRep_;
  LongVec ntTree_;

  torch::Tensor getValidActions_(
      const StackRNN& stack,
      const StackRNN& buffer,
      long num_open_NT);

  template <typename T>
  static inline std::vector<T> vecFromVar_(const torch::Tensor& v) {
    auto datatensor = v.to(torch::kCPU);
    auto dataptr = static_cast<T*>(datatensor.data_ptr());
    int datasize = datatensor.dim() == 0 ? 1 : datatensor.size(0);
    return std::vector<T>(dataptr, dataptr + datasize);
  }
};

class IntentSlotConstraints : public Parser::Constraints {
 public:
  IntentSlotConstraints() {}
  ~IntentSlotConstraints() {}

  void init(Parser& parent) override {
    auto actionsVec = parent.getActionsVec();
    for (long i = 0; i < actionsVec.size(); i++) {
      if (isUnsupportedNt_(actionsVec[i])) {
        parent.ignoreSubNtRoots.push_back(i);
      }
      if (isIntent_(actionsVec[i])) {
        parent.validNtIdxs.push_back(i);
        validIntentIdxs_.push_back(i);
      }
      if (isSlot_(actionsVec[i])) {
        parent.validNtIdxs.push_back(i);
        validSlotIdxs_.push_back(i);
      }
    }
  }

  void populateActions(Parser& parent, long lastOpenNt) override {
    // Can open intent if stack is empty or the last open NT is a slot
    if (lastOpenNt == -1 ||
        Preprocessor::hasMember(validSlotIdxs_, lastOpenNt)) {
      parent.validActions_.insert(
          parent.validActions_.end(),
          validIntentIdxs_.begin(),
          validIntentIdxs_.end());
    }
    // Can open slot if last open NT is an intent
    else if (Preprocessor::hasMember(validIntentIdxs_, lastOpenNt)) {
      parent.validActions_.insert(
          parent.validActions_.end(),
          validSlotIdxs_.begin(),
          validSlotIdxs_.end());
    }
  }

  std::pair<std::string, std::string> splitNt(std::string nt) override {
    int endOfPrefix = 0;
    if (isIntent_(nt)) {
      endOfPrefix = nt.find(kStrIntent) + kStrIntent.length();
    } else if (isSlot_(nt)) {
      endOfPrefix = nt.find(kStrSlot) + kStrSlot.length();
    }

    std::string prefix = nt.substr(0, endOfPrefix);
    std::string suffix = nt.substr(endOfPrefix);
    return std::make_pair(prefix, suffix);
  }

  static const std::string kStrIntent;
  static const std::string kStrSlot;

 protected:
  LongVec validIntentIdxs_;
  LongVec validSlotIdxs_;

  inline bool isValidNt_(const std::string& nodeLabel) const {
    return (isIntent_(nodeLabel) || isSlot_(nodeLabel));
  }
  inline bool isIntent_(const std::string& nodeLabel) const {
    auto res =
        std::mismatch(kStrIntent.begin(), kStrIntent.end(), nodeLabel.begin());
    return (res.first == kStrIntent.end());
  }
  inline bool isSlot_(const std::string& nodeLabel) const {
    auto res =
        std::mismatch(kStrSlot.begin(), kStrSlot.end(), nodeLabel.begin());
    return (res.first == kStrSlot.end());
  }
  inline bool isUnsupportedNt_(const std::string& nodeLabel) const {
    return (
        isIntent_(nodeLabel) &&
        nodeLabel.find("unsupported") != std::string::npos);
  }
};

} // namespace rnng
} // namespace assistant
} // namespace facebook
