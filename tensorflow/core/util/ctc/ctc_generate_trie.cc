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

#include "tensorflow/core/util/ctc/ctc_trie_node.h"
#include "lm/model.h"

namespace tensorflow {
namespace ctc {

typedef lm::ngram::ProbingModel Model;

// TODO this mapping needs to be provided somehow as well
char CharToVocab(char c) {
  if (c == '\'')
    return 26;
  char vocabIndex = c - 'a';
  if (vocabIndex < 0 || vocabIndex > 25)
    throw std::invalid_argument("Converted char that is not in the allowed vocabulary range");
  return vocabIndex;
}

lm::WordIndex GetWordIndex(const Model& model, const std::string& word) {
  lm::WordIndex vocab;
  vocab = model->GetVocabulary().Index(word);
  return vocab;
}

float ScoreWord(const Model& model, lm::WordIndex vocab) {
  Model::State in_state = model.NullContextState();
  Model::State out;
  lm::FullScoreReturn full_score_return;
  full_score_return = model->FullScore(in_state, vocab, out);
  return full_score_return.prob;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage " << argv[0] << " <kenlm_file_path>" << std::endl;
    return 1;
  }
  const char *kenlm_file_path = argv[1];
  lm::ngram::Config config;
  config.load_method = util::POPULATE_OR_READ;
  Model model(kenlm_file_path, config);

  TrieNode<27> root;

  std::string word;
  while (std::cin >> word) {
    lm::WordIndex vocab = GetWordIndex(word);
    float unigram_score = ScoreWord(model, vocab);
    root.Insert(word.c_str(), CharToVocab, vocab, unigram_score);
  }

  std::cout << &root;
}

}  // end namespace ctc
}  // end namespace tensorflow
