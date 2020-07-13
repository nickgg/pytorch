#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/arg_library.h>
#include <torch/csrc/jit/tensorexpr/autotuning/cpu_eval.h>
#include <torch/csrc/jit/tensorexpr/autotuning/nnc_candidates.h>
#include <torch/csrc/jit/tensorexpr/autotuning/result_library.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

class CPUTuner : public NNCTransformer<CPUCandidate> {
 public:
  CPUTuner(LoopNest* program, const std::vector<CodeGen::BufferArg>& args)
      : resultLibrary_(stats_), platform_(stats_, program, args) {
    resultLibrary_.insertCandidate(platform_.getOriginalCandidate());
  }

  void runGeneration() {
    auto beam = resultLibrary_.getBestResolvedCandidates(10);
    std::vector<CPUCandidate*> candidates;
    for (auto* c : beam) {
      if (c->children.empty()) {
        expandChildren(c);
      }

      for (auto* cc : c->children) {
        if (cc->times_run == 0) {
          candidates.push_back(dynamic_cast<CPUCandidate*>(cc));
        }
      }
    }

    platform_.runCandidates(candidates);
    resultLibrary_.insertBatch(candidates);
  }

  CPUCandidate* getBestCandidate() {
    auto vec = resultLibrary_.getBestResolvedCandidates(1);
    if (vec.empty()) {
      return nullptr;
    }
    return vec[0];
  }

  void expandChildren(CPUCandidate* c) {
    auto start = host_timestamp();
    generateSplitWithTail(c, 8);
    generateSplitWithTail(c, 16);
    generateSplitWithTail(c, 3);
    generateReorderAxis(c);
    generateRfactor(c);
    stats_.generation_time += TO_MS(host_timestamp() - start);
  }

  virtual CPUCandidate* mutateCandidate(
      std::vector<CPUCandidate*> children,
      CPUCandidate* c,
      CPUCandidate::Program* p,
      Transform* t) override {
    if (candidateHashes_.insert(resultLibrary_.hash(p->root_stmt())).second ==
        false) {
      return nullptr;
    }
    auto* newc = new CPUCandidate(p, c, t);
    children.push_back(newc);
    return newc;
  }

  const TuningStats& getStats() {
    return stats_;
  }

  std::deque<CPUCandidate*>& getAllCandidates() {
    return resultLibrary_.getAllCandidates();
  }

 protected:
  CPUPlatform platform_;
  ResultLibrary<CPUCandidate> resultLibrary_;
  std::unordered_set<SimplifierHashType> candidateHashes_;
  TuningStats stats_;
};

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch
