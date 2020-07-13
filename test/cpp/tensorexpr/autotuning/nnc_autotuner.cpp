
#include <torch/csrc/jit/tensorexpr/autotuning/gpu_tuner.h>

#include <iostream>
#include "test/cpp/tensorexpr/test_base.h"
#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"

using namespace torch::jit;
using namespace torch::jit::tensorexpr;
using namespace torch::jit::tensorexpr::tuning;

int main(void) {
  KernelScope kernel_scope;

  int M = 128;
  int N = 128;
  int K = 512;

  Buffer tX(BufHandle("tX", {M, K}, kInt));
  Buffer tA(BufHandle("tA", {M, K}, kInt));
  Buffer tB(BufHandle("tB", {K, N}, kInt));
  Buffer tY(BufHandle("tY", {M, N}, kInt));

  Tensor* ma = Compute(

      "ma",
      {{M, "m1"}, {K, "k1"}},
      [&](const ExprHandle& m, const ExprHandle& k) {
        return tX(m, k) * tA(m, k);
      });

  Tensor* mm = Reduce(
      "mm",
      {{M, "m2"}, {N, "n2"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return ma->call(m, k) * tB(k, n);
        // return tA(m, k) * tB(k, n);
      },
      {{K, "k2"}});

  Tensor* mx = Compute(
      "mx",
      {{M, "m3"}, {N, "n3"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        // return tY(m, n) + ma->call(m, n);
        return tY(m, n) + mm->call(m, n);
      });

  LoopNest loop2({ma, mm, mx});
  GPUTuner tuner(&loop2, {tX, tA, tB, tY, ma, mm, mx});
  auto* result = tuner.getBestCandidate();
  std::cout << result->times_run << " " << result->resolved_cost << "\n";
  for (int i = 1; i < 20; ++i) {
    std::cout << "Generation " << i << "\n";
    tuner.runGeneration();
  }
  result = tuner.getBestCandidate();
  std::cout << result->times_run << " " << result->resolved_cost << "\n";
  result->schedule.log();
  std::cout << "\n";
  auto stats = tuner.getStats();
  std::cout << "generated: " << stats.candidates_generated
            << " resolved: " << stats.candidates_resolved
            << " runs: " << stats.total_runs << "\n";
  std::cout << "running time: " << stats.running_time.count()
            << "ms codegen time: " << stats.codegen_time.count() << "ms\n";

  std::vector<GPUCandidate*> bestByDepth;
  bestByDepth.resize(10);
  auto& allCandidates = tuner.getAllCandidates();
  for (auto* c : allCandidates) {
    size_t depth = c->schedule.depth();
    if (depth > 10) {
      std::cout << " too big \n";
    }

    if (bestByDepth[depth] == nullptr ||
        bestByDepth[depth]->resolved_cost > c->resolved_cost) {
      bestByDepth[depth] = c;
    }
  }
  std::cout << "\n Best by depth: \n";

  for (int i = 0; i < 10; ++i) {
    std::cout << (i + 1) << ": ";
    bestByDepth[i]->schedule.log();
    std::cout << " (" << bestByDepth[i]->resolved_cost << " us)\n";
  }

  return 0;
}
