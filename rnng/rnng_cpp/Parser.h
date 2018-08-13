// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <autogradpp/autograd.h>
#include <torch/torch.h>

#include "Preprocessor.h"

namespace facebook {
namespace assistant {
namespace rnng {

using StateMap = std::map<std::string, at::Tensor>;
using StateEmbedding = std::pair<autograd::Variable, long>;
using StackElement = std::pair<autograd::Variable, StateEmbedding>;

using LongVec = std::vector<long>;
using FloatVec = std::vector<float>;
using StringVec = std::vector<std::string>;

// PredictorResult = tuple<actions, tokens, scores, pretty_print>
using PredictorResult = std::tuple<StringVec, StringVec, FloatVec, std::string>;

AUTOGRAD_CONTAINER_CLASS(Parser) {
 public:
  Parser(
      const FloatVec& modelConfig,
      // {version, lstm_dim, lstm_layers, embed_dim,
      // max_open_NT, dictfeats_embed_dim, dropout}
      const StringVec& actionsVec,
      const StringVec& terminalsVec,
      const StringVec& dictfeatsVec = std::vector<std::string>());

  void initialize_containers() override;
  void initialize_parameters() override;
  void loadStateDict(const StateMap& stateMap, bool transformNames = false);
  void initWordWeights(at::Tensor pretrainedWordWeights);
  void zeroGrad();

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
  std::vector<autograd::Variable> forward(std::vector<autograd::Variable> x)
      override;

  static const std::string kStrShift;
  static const std::string kStrReduce;
  static const std::string kStrOpenBracket;
  static const std::string kStrCloseBracket;

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
        autograd::Container lstm,
        const autograd::Variable& state,
        const autograd::Variable& emptyEmbedding);

    autograd::Variable embedding();
    long elementFromTop(long index) const;
    void pop();
    void push(const autograd::Variable& expression, long token);
    std::pair<autograd::Variable, long> top() const;

    inline size_t size() const {
      return stack_.size() - 1; // Ignoring empty embedding
    }

   private:
    Parser& parent_;
    autograd::Container lstm_;
    autograd::Variable emptyEmbedding_;
    std::vector<StackElement> stack_;
    autograd::Variable rnnGetOutput_(const autograd::Variable& state) const;
  };

  Preprocessor preprocessor_;
  std::shared_ptr<Constraints> constraints_;

  autograd::Container wordsLookup_;
  autograd::Container actionsLookup_;
  autograd::Container dictfeatsLookup_;
  autograd::Container actionLinear0_;
  autograd::Container actionLinear1_;
  autograd::Container dropoutLayer_;
  autograd::Container bufferRnn_;
  autograd::Container stackRnn_;
  autograd::Container actionRnn_;
  autograd::Container compositional_;
  autograd::Variable emptyBufferEmb_;
  autograd::Variable emptyStackEmb_;
  autograd::Variable emptyActionEmb_;
  at::Tensor actionP_;

  bool addDictFeats_ = false;
  long lstmDim_;
  long lstmLayers_;
  long embedDim_;
  long maxOpenNt_;
  long dictfeatsEmbedDim_;
  float dropout_;

  StringVec actionsVec_;
  StringVec terminalsVec_;
  StringVec dictfeatsVec_;
  long shiftIdx_;
  long reduceIdx_;
  long compositionalIdx_;

  LongVec predictedActionsIdx_;
  std::vector<bool> isOpenNt_;
  std::vector<at::Tensor> poppedRep_;
  LongVec ntTree_;

  autograd::Variable xavierInitialState_(const at::IntList& dims);
  autograd::Variable getValidActions_(
      const StackRNN& stack, const StackRNN& buffer, long num_open_NT);

  template <typename T>
  inline autograd::Variable varFromVec_(
      const std::vector<T>& v, at::ScalarType scalarType) {
    // Cannot deduce scalarType from template because ATen defines macros for an
    // enum class
    auto tempTensor =
        at::CPU(scalarType)
            .tensorFromBlob(
                (void*)&(v.data())[0], {static_cast<long>(v.size())})
            .clone();
    if (cuda_) {
      tempTensor = tempTensor.toBackend(at::kCUDA);
    }
    return autograd::Var(tempTensor, false);
  }

  template <typename T>
  static inline std::vector<T> vecFromVar_(const autograd::Variable& v) {
    auto datatensor = v.toBackend(at::kCPU);
    auto dataptr = static_cast<T*>(datatensor.data_ptr());
    int datasize = datatensor.dim() == 0 ? 1 : datatensor.size(0);
    return std::vector<T>(dataptr, dataptr + datasize);
  }

  // Adaptation of PyTorch's _calculate_fan_in_and_fan_out
  static long fanInOut_(const at::IntList& dims);
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
