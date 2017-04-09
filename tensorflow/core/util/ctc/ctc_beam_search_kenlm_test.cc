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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ctc/ctc_beam_entry.h"
#include "tensorflow/core/util/ctc/ctc_beam_scorer.h"

namespace {

using tensorflow::ctc::KenLMBeamScorer;
using tensorflow::ctc::LabelToCharacterTranslator;
using tensorflow::ctc::ctc_beam_search::KenLMBeamState;

const char test_sentence[] = "tomorrow it will rain";
// Input path for 'tomorrow it will rain'
const int test_labels_count = 108;
const int test_labels[] = {19,19,19,19,28,28,14,28,28,12,12,12,28,14,14,14,14,28,
                      28,17,17,28,28,28,17,17,17,17,28,14,14,14,28,28,28,28,
                      22,22,22,22,28,28,28,27,27,27,27,28,28,28,28,8,8,28,28,
                      28,19,19,19,28,28,28,27,28,22,22,22,28,28,28,8,28,28,28,
                      11,11,11,11,28,11,11,28,28,27,27,27,28,28,17,28,28,28,
                      28,0,0,28,28,28,8,8,28,28,28,13,13,13,13,28};
// Input path for 'tomorrow it will rain th'
const int test_labels_incomplete_count = 110;
const int test_labels_incomplete[] = {19,19,19,19,28,28,14,28,28,12,12,12,28,14,14,14,14,28,
                      28,17,17,28,28,28,17,17,17,17,28,14,14,14,28,28,28,28,
                      22,22,22,22,28,28,28,27,27,27,27,28,28,28,28,8,8,28,28,
                      28,19,19,19,28,28,28,27,28,22,22,22,28,28,28,8,28,28,28,
                      11,11,11,11,28,11,11,28,28,27,27,27,28,28,17,28,28,28,
                      28,0,0,28,28,28,8,8,28,28,28,13,13,13,13,28,19,7};
// Input path for 'tomorow it will rain'
const int test_labels_typo_count = 106;
const int test_labels_typo[] = {19,19,19,19,28,28,14,28,28,12,12,12,28,14,14,14,14,28,
                      28,28,28,28,17,17,17,17,28,14,14,14,28,28,28,28,
                      22,22,22,22,28,28,28,27,27,27,27,28,28,28,28,8,8,28,28,
                      28,19,19,19,28,28,28,27,28,22,22,22,28,28,28,8,28,28,28,
                      11,11,11,11,28,11,11,28,28,27,27,27,28,28,17,28,28,28,
                      28,0,0,28,28,28,8,8,28,28,28,13,13,13,13,28};

KenLMBeamScorer *createKenLMBeamScorer() {
  const char *kenlm_file_path = "./tensorflow/core/util/ctc/testdata/testing-kenlm.binary";
  return new KenLMBeamScorer(kenlm_file_path);
}

TEST(LabelToCharacterTranslator, GetCharacterFromLabel) {

  LabelToCharacterTranslator translator;

  int previous_label = 0;
  int test_sentence_offset = 0;
  for (int i = 0; i < test_labels_count; i++) {
    int label = test_labels[i];
    if (label != previous_label && !translator.IsBlankLabel(label)) {
      char returned_char = translator.GetCharacterFromLabel(label);
      EXPECT_EQ(test_sentence[test_sentence_offset++], returned_char);
    }
    previous_label = label;
  }
}

float ScoreBeam(KenLMBeamScorer *scorer, const int labels[], const int label_count) {
  KenLMBeamState states[2];
  scorer->InitializeState(&states[0]);

  int from_label = -1;
  for (int i = 0; i < label_count; i++) {
    int to_label = labels[i];
    
    scorer->ExpandState(states[i % 2], from_label,
                        &states[(i + 1) % 2], to_label);
    
    // Update from_label for next iteration
    from_label = to_label;
  }

  KenLMBeamState &endState = states[label_count % 2];
  scorer->ExpandStateEnd(&endState);

  float log_prob = scorer->GetStateEndExpansionScore(endState);

  return log_prob;
}

TEST(KenLMBeamSearch, PenalizeIncompleteWord) {
  KenLMBeamScorer *scorer = createKenLMBeamScorer();

  float log_prob_sound = ScoreBeam(scorer, test_labels, test_labels_count);
  float log_prob_incomplete = ScoreBeam(scorer, test_labels_incomplete, test_labels_incomplete_count);

  delete scorer;

  EXPECT_GT(log_prob_sound, log_prob_incomplete);
}

TEST(KenLMBeamSearch, PenalizeTypos) {
  KenLMBeamScorer *scorer = createKenLMBeamScorer();

  float log_prob_sound = ScoreBeam(scorer, test_labels, test_labels_count);
  float log_prob_typo = ScoreBeam(scorer, test_labels_typo, test_labels_typo_count);

  delete scorer;

  EXPECT_GT(log_prob_sound, log_prob_typo);
}

TEST(KenLMBeamSearch, ExpandState) {
  KenLMBeamScorer *scorer = createKenLMBeamScorer();

  float log_prob = ScoreBeam(scorer, test_labels, test_labels_count);

  delete scorer;

  EXPECT_NEAR(-4.21812, log_prob, 0.0001);
}

}  // namespace
