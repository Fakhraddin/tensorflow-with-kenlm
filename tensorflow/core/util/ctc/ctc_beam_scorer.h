/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Collection of scoring classes that can be extended and provided to the
// CTCBeamSearchDecoder to incorporate additional scoring logic (such as a
// language model).
//
// To build a custom scorer extend and implement the pure virtual methods from
// BeamScorerInterface. The default CTC decoding behavior is implemented
// through BaseBeamScorer.

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_

#include "tensorflow/core/util/ctc/ctc_beam_entry.h"
#include "lm/model.hh"

namespace tensorflow {
namespace ctc {

using namespace ctc_beam_search;

// Base implementation of a beam scorer used by default by the decoder that can
// be subclassed and provided as an argument to CTCBeamSearchDecoder, if complex
// scoring is required. Its main purpose is to provide a thin layer for
// integrating language model scoring easily.
template <typename CTCBeamState>
class BaseBeamScorer {
 public:
  virtual ~BaseBeamScorer() {}
  // State initialization.
  virtual void InitializeState(CTCBeamState* root) const {}
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  virtual void ExpandState(const CTCBeamState& from_state, int from_label,
                           CTCBeamState* to_state, int to_label) const {}
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  virtual void ExpandStateEnd(CTCBeamState* state) const {}
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  virtual float GetStateExpansionScore(const CTCBeamState& state,
                                       float previous_score) const {
    return previous_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  virtual float GetStateEndExpansionScore(const CTCBeamState& state) const {
    return 0;
  }
};

class KenLMBeamScorer : public BaseBeamScorer<KenLMBeamState> {
 public:
  typedef lm::ngram::ProbingModel Model;

  virtual ~KenLMBeamScorer() {
    delete model;
  }
  KenLMBeamScorer(const char *kenlm_file_path) {
    lm::ngram::Config config;
    config.load_method = util::POPULATE_OR_READ;

    model = new Model(kenlm_file_path, config);
  }

  // State initialization.
  void InitializeState(KenLMBeamState* root) const {
    root->complete_words_score = 0.0f;
    root->incomplete_word_score = 0.0f;
    root->incomplete_word.clear();
    root->model_state = model->BeginSentenceState();
  }
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  void ExpandState(const KenLMBeamState& from_state, int from_label,
                           KenLMBeamState* to_state, int to_label) const {
    Model::State out;

    CopyState(from_state, to_state);

    if (from_label == to_label || IsBlankLabel(to_label)) {
      return;
    }

    if (!IsSpaceLabel(to_label)) {
      to_state->incomplete_word += GetCharacterFromLabel(to_label);
    }

    float prob = ScoreIncompleteWord(from_state.model_state,
                          to_state->incomplete_word,
                          out);
    to_state->incomplete_word_score = prob;

    if (IsSpaceLabel(to_label)) {
      to_state->complete_words_score += to_state->incomplete_word_score;
      to_state->incomplete_word_score = 0.0f;
      to_state->incomplete_word.clear();
      to_state->model_state = out;
    }
  }
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  void ExpandStateEnd(KenLMBeamState* state) const {
    Model::State out;
    lm::FullScoreReturn ret;
    if (state->incomplete_word.size() > 0) {
      float prob = ScoreIncompleteWord(state->model_state,
                                        state->incomplete_word,
                                        out);
      state->complete_words_score += prob;
      state->incomplete_word_score = 0.0f;
      state->incomplete_word.clear();
      state->model_state = out;
    }
    ret = model->FullScore(state->model_state,
                            model->GetVocabulary().EndSentence(),
                            out);
    state->complete_words_score += ret.prob;
  }
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  float GetStateExpansionScore(const KenLMBeamState& state,
                                       float previous_score) const {
    return state.complete_words_score + state.incomplete_word_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  float GetStateEndExpansionScore(const KenLMBeamState& state) const {
    return state.complete_words_score;
  }

 private:
  Model *model;

  float ScoreIncompleteWord(const Model::State& model_state,
                            const std::string& word,
                            Model::State& out) const {
    lm::FullScoreReturn ret;
    lm::WordIndex vocab;
    vocab = model->GetVocabulary().Index(word);
    ret = model->FullScore(model_state, vocab, out);
    return ret.prob;
  }

  void CopyState(const KenLMBeamState& from, KenLMBeamState* to) const {
    to->complete_words_score = from.complete_words_score;
    to->incomplete_word_score = from.incomplete_word_score;
    to->incomplete_word = from.incomplete_word;
    to->model_state = from.model_state;
  }

  bool inline IsBlankLabel(int label) const {
    return label == 29;
  }

  bool inline IsSpaceLabel(int label) const {
    return label == 28;
  }

  char GetCharacterFromLabel(int label) const {
    if (label == 27) {
      return '\'';
    }
    if (label == 28) {
      return ' ';
    }
    return label + 'a';
  }

};

}  // namespace ctc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
