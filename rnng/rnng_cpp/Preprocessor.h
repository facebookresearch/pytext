// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <autogradpp/autograd.h>
#include <re2/re2.h>
#include <cctype>

namespace facebook {
namespace assistant {
namespace rnng {
using StringVec = std::vector<std::string>;

class Preprocessor {
 public:
  static const std::string kStrUnk;
  static const std::string kStrNum;

  // Adaptation of github.com/clab/rnng/blob/master/get_oracle.p
  static std::string unkifyToken(
      const std::string& rawToken,
      const StringVec& words);

  static std::vector<long> getIndices(
      const StringVec& rawItems,
      const StringVec& vocab,
      bool addDictFeat = false);

  template <typename T>
  static long findIdx(const std::vector<T>& vec, const T& item) {
    return std::find(vec.begin(), vec.end(), item) - vec.begin();
  }

  template <typename T>
  static bool hasMember(const std::vector<T>& vec, const T& item) {
    return findIdx(vec, item) != vec.size();
  }

  static inline bool hasSuffix(
      const std::string& str,
      const std::string& suffix) {
    return str.size() >= suffix.size() &&
        str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
  }
};

} // namespace rnng
} // namespace assistant
} // namespace facebook
