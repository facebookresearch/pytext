// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <cassert>
#include <set>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <glog/logging.h>

using namespace std;
using json = nlohmann::json;

class Formatter {
 public:
  const string& formatRequest(const string& requestBody) {
    const json jsonRequest = json::parse(requestBody);

    string text;
    try {
      text = jsonRequest.at(mTextParam);
      LOG(INFO) << "Requested processing of \"" << text << "\"";
    } catch (json::out_of_range e) {
      throw out_of_range(e.what());
    }

    string normalizedText = text;
    transform(text.begin(), text.end(), normalizedText.begin(), ::tolower);
    normalizedText = stripPrefixChars(normalizedText, mPrefixCharsToStrip);
    normalizedText = stripSuffixChars(normalizedText, mSuffixCharsToStrip);
    VLOG(1) << "Normalized \"" << text << "\" into \"" << normalizedText << "\"";

    return normalizedText;
  }

  const string& formatResponse(const map<string, double>& scores, const string& text) {
    // Exponentiate
    map<string, double> expScores;
    transform(scores.begin(), scores.end(), inserter(expScores, expScores.begin()),
              [](const auto& p) {
                return make_pair(p.first, exp(p.second));
              });

    // Sum up
    double sum = accumulate(begin(expScores), end(expScores), 0.,
                            [](double previous, const auto& p) { return previous + p.second; });

    // Normalize (end up with softmax)
    map<string, double> normScores;
    transform(expScores.begin(), expScores.end(), inserter(normScores, normScores.begin()),
              [sum](const auto& p) {
                return make_pair(p.first, p.second / sum);
              });

    // Sort in descending order
    vector<pair<string, double>> sortedScores = sortMapByValue(normScores);
    VLOG(1) << "Normalized scores for \"" << text << "\": " << normScores;

    // Reformat into name / confidence pairs. Strip "intent:" prefix
    json ir = json::array();
    for (const auto& p : sortedScores) {
      ir.push_back({{mName, stripPrefixWord(p.first, mIntentPrefix)},
                    {mConfidence, p.second}});
    }

    json j;
    j[mText] = text;
    j[mIntentRanking] = ir;
    if (!ir.empty()) {
      j[mIntent] = ir.at(0);
    } else {
      j[mIntent] = nullptr;
    }
    j[mEntities] = json::array();

    LOG(INFO) << "Processed \"" << text << "\", predicted " << (j[mIntent] != nullptr ? j[mIntent].dump() : "no intents");
    return j.dump(2 /*indentation*/);
  }

  template <typename A, typename B>
  vector<pair<A, B>> sortMapByValue(const map<A, B>& src) {
    vector<pair<A, B>> v{make_move_iterator(begin(src)),
                         make_move_iterator(end(src))};

    sort(begin(v), end(v),
         [](auto lhs, auto rhs) { return lhs.second > rhs.second; });  // descending order

    return v;
  }

  const string& stripPrefixWord(const string& text, const string& prefix) {
    if (text.length() >= prefix.length()) {
      auto res = std::mismatch(prefix.begin(), prefix.end(), text.begin());
      if (res.first == prefix.end()) {
        return text.substr(prefix.length());
      }
    }
    return text;
  }

  const string& stripPrefixChars(const string& text, const set<char>& prefixes) {
    int textLength = text.length();  // cast to int to avoid unsigned size_t underflow on subtraction from 0
    int startIdx = 0;
    while (startIdx < textLength) {
      if (prefixes.find(text.at(startIdx)) == prefixes.end()) { // character is not in prefixes set
        break;
      }
      startIdx++;
    }
    return text.substr(startIdx);
  }

  const string& stripSuffixWord(const string& text, const string& suffix) {
    if (text.length() >= suffix.length()) {
      if (0 == text.compare(text.length() - suffix.length(), suffix.length(), suffix)) {
        return text.substr(0, text.length() - suffix.length());
      }
    }
    return text;
  }

  const string& stripSuffixChars(const string& text, const set<char>& suffixes) {
    int endIdx = text.length();  // cast to int to avoid unsigned size_t underflow on subtraction from 0
    while (endIdx > 0) {
      if (suffixes.find(text.at(endIdx - 1)) == suffixes.end()) { // character is not in suffixes set
        break;
      }
      endIdx--;
    }
    return text.substr(0, endIdx);
  }

  void runTests() {
    assert(stripPrefixWord("foo:bar", "") == "foo:bar");
    assert(stripPrefixWord("foo:bar", "foo:") == "bar");
    assert(stripPrefixWord("foo:bar", "food") == "foo:bar");
    assert(stripPrefixWord("foo:", "foo:bar") == "foo:");
    assert(stripPrefixWord("", "foo:") == "");

    assert(stripPrefixChars("foo bar", {' ', '?'}) == "foo bar");
    assert(stripPrefixChars("?foo bar", {' ', '?'}) == "foo bar");
    assert(stripPrefixChars("??  foo bar", {' ', '?'}) == "foo bar");
    assert(stripPrefixChars("?! ?foo bar", {' ', '?', '!'}) == "foo bar");
    assert(stripPrefixChars(" !?!?!  ", {' ', '?', '!'}) == "");
    assert(stripPrefixChars("", {' ', '?', '!'}) == "");

    assert(stripSuffixWord("foo:bar", "") == "foo:bar");
    assert(stripSuffixWord("foo:bar", "bar") == "foo:");
    assert(stripSuffixWord("foo:bar", "bars") == "foo:bar");
    assert(stripSuffixWord("foo:", "foo:bar") == "foo:");
    assert(stripSuffixWord("", "bar") == "");

    assert(stripSuffixChars("foo bar", {' ', '?'}) == "foo bar");
    assert(stripSuffixChars("foo bar?", {' ', '?'}) == "foo bar");
    assert(stripSuffixChars("foo bar  ??", {' ', '?'}) == "foo bar");
    assert(stripSuffixChars("foo bar? !? ", {' ', '?', '!'}) == "foo bar");
    assert(stripSuffixChars(" !?!?!  ", {' ', '?', '!'}) == "");
    assert(stripSuffixChars("", {' ', '?', '!'}) == "");

    // Composite
    assert (
      stripSuffixChars(
        stripPrefixChars(" what is foo.?!?? ", mPrefixCharsToStrip),
        mSuffixCharsToStrip
      ) == "what is foo"
    );

    LOG(INFO) << "All formatter tests passed";
  }

  static const string mTextParam;
  static const string mName;
  static const string mConfidence;
  static const string mIntentPrefix;
  static const string mText;
  static const string mIntentRanking;
  static const string mIntent;
  static const string mEntities;

  static const set<char> mPrefixCharsToStrip;
  static const set<char> mSuffixCharsToStrip;
};

const string Formatter::mTextParam = "text";
const string Formatter::mName = "name";
const string Formatter::mConfidence = "confidence";
const string Formatter::mIntentPrefix = "intent:";
const string Formatter::mText = "text";
const string Formatter::mIntentRanking = "intent_ranking";
const string Formatter::mIntent = "intent";
const string Formatter::mEntities = "entities";

const set<char> Formatter::mPrefixCharsToStrip = {' '};
const set<char> Formatter::mSuffixCharsToStrip = {' ', '?', '.', '!'};
