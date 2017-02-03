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
using tensorflow::ctc::ctc_beam_search::KenLMBeamState;

TEST(CtcBeamSearch, KenMLBeamSearch) {

  const char *kenlm_file_path = "";
  KenLMBeamScorer scorer(kenlm_file_path);

  // Input path for 'tomorrow it will rain'
  const int labelCount = 108;
  const int labels[] = {19,19,19,19,29,29,14,29,29,12,12,12,29,14,14,14,14,29,
                        29,17,17,29,29,29,17,17,17,17,29,14,14,14,29,29,29,29,
                        22,22,22,22,29,29,29,28,28,28,28,29,29,29,29,8,8,29,29,
                        29,19,19,19,29,29,29,28,29,22,22,22,29,29,29,8,29,29,29,
                        11,11,11,11,29,11,11,29,29,28,28,28,29,29,17,29,29,29,
                        29,0,0,29,29,29,8,8,29,29,29,13,13,13,13,29};

  KenLMBeamState states[2];
  scorer.InitializeState(&states[0]);

  int from_label = -1;
  for (int i = 0; i < labelCount; i++) {
    int to_label = labels[i];
    
    scorer.ExpandState(states[i % 2], from_label,
                        &states[i % 2 + 1], to_label);
    
    // Update from_label for next iteration
    from_label = to_label;
  }

  KenLMBeamState &endState = states[labelCount % 2];
  scorer.ExpandStateEnd(&endState);

  float log_prob = scorer.GetStateEndExpansionScore(endState);

  EXPECT_EQ(log_prob, -14.8305);
}

}  // namespace
