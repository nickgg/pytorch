#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/arg_library.h>
#include <torch/csrc/jit/tensorexpr/autotuning/gpu_eval.h>
#include <torch/csrc/jit/tensorexpr/autotuning/nnc_candidates.h>
#include <torch/csrc/jit/tensorexpr/autotuning/result_library.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

class GPUTuner : public NNCTransformer<GPUCandidate> {
 public:
  GPUTuner(LoopNest* program, const std::vector<CodeGen::BufferArg>& args)
      : resultLibrary_(stats_), platform_(stats_, program, args) {
    resultLibrary_.insertCandidate(platform_.getOriginalCandidate());
  }

  void runGeneration() {
    auto beam = resultLibrary_.getBestResolvedCandidates(10);
    std::vector<GPUCandidate*> candidates;
    for (auto* c : beam) {
      if (c->children.empty()) {
        expandChildren(c);
      }

      for (auto* cc : c->children) {
        if (cc->times_run == 0 && candidates.size() < 50) {
          candidates.push_back(dynamic_cast<GPUCandidate*>(cc));
        }
      }
    }

    platform_.runCandidates(candidates);
    resultLibrary_.insertBatch(candidates);
  }

  GPUCandidate* getBestCandidate() {
    auto vec = resultLibrary_.getBestResolvedCandidates(1);
    if (vec.empty()) {
      return nullptr;
    }
    return vec[0];
  }

  void expandChildren(GPUCandidate* c) {
    auto start = host_timestamp();
    generateSplitWithMask(c, 8);
    generateSplitWithMask(c, 3);
    mutateSplitWithMask(c);
    generateReorderAxis(c);
    generateRfactor(c);
    generateNextAxisBinding(c);
    mutateAxisBinding(c);
    stats_.generation_time += TO_MS(host_timestamp() - start);
  }

  virtual GPUCandidate* mutateCandidate(
      std::vector<GPUCandidate*> children,
      GPUCandidate* c,
      GPUCandidate::Program* p,
      Transform* t) override {
    if (candidateHashes_.insert(resultLibrary_.hash(p->root_stmt())).second ==
        false) {
      return nullptr;
    }
    auto* newc = new GPUCandidate(p, c, t);
    children.push_back(newc);
    return newc;
  }

  const TuningStats& getStats() {
    return stats_;
  }

  std::deque<GPUCandidate*>& getAllCandidates() {
    return resultLibrary_.getAllCandidates();
  }

 protected:
  GPUPlatform platform_;
  ResultLibrary<GPUCandidate> resultLibrary_;
  std::unordered_set<SimplifierHashType> candidateHashes_;
  TuningStats stats_;
};

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch
