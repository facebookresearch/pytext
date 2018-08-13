// Copyright 2004-present Facebook. All Rights Reserved.

#include "Preprocessor.h"

namespace facebook {
namespace assistant {
namespace rnng {

const std::string Preprocessor::kStrUnk = "UNK";
const std::string Preprocessor::kStrNum = "NUM";

std::vector<long> Preprocessor::getIndices(
    const StringVec& rawItems,
    const StringVec& vocab,
    bool addDictFeat) {

  // Unkify if needed
  StringVec fixedItems;
  if (!addDictFeat) {
    std::transform(
        rawItems.begin(),
        rawItems.end(),
        back_inserter(fixedItems),
        [&](const std::string& rawToken) {
          return unkifyToken(rawToken, vocab);
        });
  } else {
    fixedItems = rawItems;
  }

  // Look up item in vocab. If not found, look up UNK
  std::vector<long> vocabIdxs;
  std::transform(
      fixedItems.begin(),
      fixedItems.end(),
      back_inserter(vocabIdxs),
      [&](const std::string& unkifiedItem) {
        long idx = findIdx(vocab, unkifiedItem);
        if (idx == vocab.size()) {
          idx = findIdx(vocab, kStrUnk);
        }
        return idx;
      });
  return vocabIdxs;
}

std::string Preprocessor::unkifyToken(
    const std::string& rawToken,
    const StringVec& words) {
  std::string token = rawToken;
  RE2 rgxWhitespace("\\s+$");
  RE2::Replace(&token, rgxWhitespace, "");

  if (token.length() == 0) {
    return kStrUnk;
  }
  if (std::find(words.begin(), words.end(), token) != words.end()) {
    return token;
  }

  int numCaps = 0;
  bool hasDigit = false;
  bool hasDash = false;
  bool hasLower = false;

  for (char& c : token) {
    hasDigit = hasDigit || isdigit(c);
    hasDash = hasDash || (c == '-');
    if (isalpha(c)) {
      hasLower = hasLower || islower(c);
      numCaps += isupper(c) ? 1 : 0;
    }
  }

  std::stringstream result;
  result << kStrUnk;

  char ch0 = token[0];
  std::transform(token.begin(), token.end(), token.begin(), tolower);
  if (isupper(ch0)) {
    if (numCaps == 1) {
      result << "-INITC";
      if (std::find(words.begin(), words.end(), token) != words.end()) {
        return "-KNOWNLC";
      }
    } else {
      result << "-CAPS";
    }
  } else if (!isalpha(ch0) && numCaps) {
    result << "-CAPS";
  } else if (hasLower) {
    result << "-LC";
  }

  if (hasDigit) {
    result << "-kStrNum";
  }
  if (hasDash) {
    result << "-DASH";
  }

  if (hasSuffix(token, "s") && token.length() >= 3) {
    char ch2 = token[token.length() - 2];
    if (ch2 != 's' && ch2 != 'i' && ch2 != 'u') {
      result << "-s";
    }
  } else if (token.length() >= 5 && !hasDash && !(hasDigit && numCaps)) {
    StringVec suffices = {
        "ed", "ing", "ion", "er", "est", "ly", "ity", "y", "al"};
    for (const auto& suffix : suffices) {
      if (hasSuffix(token, suffix)) {
        result << "-" << suffix;
      }
    }
  }

  return result.str();
}

} // namespace rnng
} // namespace assistant
} // namespace facebook
