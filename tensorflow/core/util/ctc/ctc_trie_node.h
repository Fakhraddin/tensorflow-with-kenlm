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


#include <functional>
#include <istream>
#include <iostream>

namespace tensorflow {
namespace ctc {

template <int VOCAB_SIZE>
class TrieNode {
public:
  TrieNode() : prefixCount(0), children{nullptr} {}

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
    os << obj->prefixCount << std::endl;
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
    obj->prefixCount = prefixCount;
    for (int i = 0; i < VOCAB_SIZE; i++) {
      // Recursive call
      is >> obj->children[i];
    }
    return is;
  }

  void Insert(const char* word, std::function<char (char)> translator) {
    char wordCharacter = *word;
    prefixCount++;
    if (wordCharacter != '\0') {
      char vocabIndex = translator(wordCharacter);
      TrieNode *child = children[vocabIndex];
      if (child == nullptr)
        child = children[vocabIndex] = new TrieNode();
      child->Insert(word + 1, translator);
    }
  }

  int GetFrequencyOf(const char* word, std::function<char (char)> translator) {
    char wordCharacter = *word;
    if (wordCharacter != '\0') {
      char vocabIndex = translator(wordCharacter);
      TrieNode *child = children[vocabIndex];
      if (child == nullptr)
        return 0;
      return child->GetFrequencyOf(word + 1, translator);
    }
    return prefixCount;
  }
  
  TrieNode *GetChildAt(int vocabIndex) {
    return children[vocabIndex];
  }

private:
  int prefixCount;
  TrieNode *children[VOCAB_SIZE];
};

} // namespace ctc
} // namespace tensorflow

#endif //CTC_TRIENODE_H
