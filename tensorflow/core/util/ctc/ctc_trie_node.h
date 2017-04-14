/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef CTC_TRIENODE_H
#define CTC_TRIENODE_H

#include "lm/model.hh"

#include <functional>
#include <istream>
#include <iostream>
#include <limits>

namespace tensorflow {
namespace ctc {

template <int VOCAB_SIZE>
class TrieNode {
public:
  TrieNode() : prefixCount(0),
                min_score_word(0),
                min_unigram_score(std::numeric_limits<float>::max()),
                children{nullptr} {}

  ~TrieNode() {
    for (int i = 0; i < VOCAB_SIZE; i++) {
      delete children[i];
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const TrieNode* obj) {
    if (obj == nullptr) {
      os << -1 << std::endl;
      return os;
    }
    obj->WriteNode(os);
    for (int i = 0; i < VOCAB_SIZE; i++) {
      // Recursive call
      os << obj->children[i];
    }
    return os;
  }

  friend std::istream& operator>>(std::istream& is, TrieNode* &obj) {
    int prefixCount;
    is >> prefixCount;

    if (prefixCount == -1) {
      // This is an undefined child
      obj = nullptr;
      return is;
    }

    obj = new TrieNode;
    obj->ReadNode(is, prefixCount);
    for (int i = 0; i < VOCAB_SIZE; i++) {
      // Recursive call
      is >> obj->children[i];
    }
    return is;
  }

  void Insert(const char* word, std::function<char (char)> translator,
              lm::WordIndex lm_word, float unigram_score) {
    char wordCharacter = *word;
    prefixCount++;
    if (unigram_score < min_unigram_score) {
      min_unigram_score = unigram_score;
      min_score_word = lm_word;
    }
    if (wordCharacter != '\0') {
      char vocabIndex = translator(wordCharacter);
      TrieNode *child = children[vocabIndex];
      if (child == nullptr)
        child = children[vocabIndex] = new TrieNode();
      child->Insert(word + 1, translator, lm_word, unigram_score);
    }
  }

  int GetFrequency() {
    return prefixCount;
  }

  lm::WordIndex GetMinScoreWordIndex() {
    return min_score_word;
  }

  float GetMinUnigramScore() {
    return min_unigram_score;
  }
  
  TrieNode *GetChildAt(int vocabIndex) {
    return children[vocabIndex];
  }

private:
  int prefixCount;
  lm::WordIndex min_score_word;
  float min_unigram_score;
  TrieNode *children[VOCAB_SIZE];

  void WriteNode(std::ostream& os) const {
    os << prefixCount << std::endl;
    os << min_score_word << std::endl;
    os << min_unigram_score << std::endl;
  }

  void ReadNode(std::istream& is, int first_input) {
    prefixCount = first_input;
    is >> min_score_word;
    is >> min_unigram_score;
  }

};

} // namespace ctc
} // namespace tensorflow

#endif //CTC_TRIENODE_H
