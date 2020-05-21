#include "test/cpp/tensorexpr/test_base.h"
#include "test/cpp/tensorexpr/test_utils.h"

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include "torch/csrc/jit/tensorexpr/auto_tuner.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"

#include <cmath>

namespace torch {
namespace jit {

void doKernelHeuristicOpts(std::vector<Tensor*> tensors, LoopNest& l) {
  for (auto* p : tensors) {
    l.computeInline(p->buf());
  }

  l.prepareForCodegen();

  std::vector<For*> innerLoops;
  std::vector<For*> worklist;

  // Find outer-most For loops
  if (For* rootF = dynamic_cast<For*>(l.root_stmt())) {
    worklist.push_back(rootF);
  } else if (Block* body = dynamic_cast<Block*>(l.root_stmt())) {
    std::vector<Block*> blocks = {body};
    while (blocks.size()) {
      Block* b = blocks.back();
      blocks.pop_back();

      for (Stmt* s : *b) {
        if (For* f = dynamic_cast<For*>(s)) {
          worklist.push_back(f);
        } else if (Block* b2 = dynamic_cast<Block*>(s)) {
          blocks.push_back(b2);
        }
      }
    }

    // Traverse the For loop nest find inner - most loops,
    //     which are vectorization candidates.
    while (worklist.size()) {
      For* f = worklist.back();
      worklist.pop_back();

      bool containsSubLoops = false;
      if (Block* body = dynamic_cast<Block*>(f->body())) {
        for (Stmt* s2 : *body) {
          if (For* f2 = dynamic_cast<For*>(s2)) {
            containsSubLoops = true;
            worklist.push_back(f2);
          }
        }
      }

      if (!containsSubLoops) {
        innerLoops.push_back(f);
      }
    }

    // vectorize inner loops.
    for (For* loop : innerLoops) {
      For* outer1;
      For* split1;
      For* tail1;

      l.splitWithTail(loop, 8, &outer1, &split1, &tail1);
      // l.vectorize(split1);

      if (tail1) {
        For* outer2;
        For* split2;
        For* tail2;
        l.splitWithTail(tail1, 4, &outer2, &split2, &tail2);
        // l.vectorize(split2);
      }
    }
  }
}

void testAutotuningSimple() {
  KernelScope kernel_scope;

  // int M = 9;
  // int N = 7;
  // int K = 12;
  // int M = 10;
  // int N = 1;
  // int K = 1000000;
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

  std::vector<int> tX_(M * K);
  std::vector<int> tA_(M * K);
  std::vector<int> tB_(N * K);
  std::vector<int> tY_(M * N);

  // std::vector<int> ma_(M * K, -1.f);
  // std::vector<int> mm_(M * N, -1.f);

  std::vector<int> out1(M * N, -1);
  std::vector<int> out2(M * N, -1);
  std::vector<int> out3(M * N, -1);

  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      tX_[m * K + k] = m * K + k;
      tA_[m * K + k] = m * K + k;
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      tB_[k * N + n] = k * N + n;
    }
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      tY_[m * N + n] = m * N + n;
    }
  }

  LoopNest loop({mx});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  LLVMCodeGen cg1(loop.root_stmt(), {tX, tA, tB, tY, mx});

  std::vector<CodeGen::CallArg> args({tX_, tA_, tB_, tY_, out1});

  auto start = std::chrono::high_resolution_clock::now();
  cg1.call(args);
  auto end = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Pre tune ran in: " << dur.count() << " us\n";

  LoopNest kOptLoop({mx});
  doKernelHeuristicOpts({ma}, kOptLoop);

  Stmt* s3 = IRSimplifier::simplify(kOptLoop.root_stmt());
  std::cout << "GOLDEN " << *s3 << "\n";
  LLVMCodeGen cg3(s3, {tX, tA, tB, tY, mx});
  std::chrono::microseconds total(0);

  for (int i = 0; i < 10; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    cg3.call({tX_, tA_, tB_, tY_, out3});
    auto end = std::chrono::high_resolution_clock::now();
    total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }
  std::cout << "Golden ref ran in: " << (total / 10).count() << " us\n";

  for (int i = 0; i < N * M; ++i) {
    ASSERT_NEAR(out1[i], out3[i], 0.01);
  }

  std::cout << "warming up\n";
  for (int i = 0; i < 10; ++i) {
    cg1.call(args);
  }

  LoopNest loop2({mx});
  AutoTuner tuner(loop2, {tX, tA, tB, tY, mx});
  tuner.run(10);

  LoopNest tunedLoop = tuner.getBestCandidate();
  tunedLoop.prepareForCodegen();
  Stmt* s2 = tunedLoop.root_stmt();
  s2 = IRSimplifier::simplify(s2);

  LLVMCodeGen cg2(s2, {tX, tA, tB, tY, mx});

  start = std::chrono::high_resolution_clock::now();
  end = std::chrono::high_resolution_clock::now();
  dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::chrono::microseconds total2(0);

  for (int i = 0; i < 10; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    cg2.call({tX_, tA_, tB_, tY_, out2});
    auto end = std::chrono::high_resolution_clock::now();
    total2 +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }
  std::cout << "Pos tune ran in: " << (total2 / 10).count() << " us\n";
  for (int i = 0; i < N * M; ++i) {
    // if (i < 10) {
    //   std::cout << out1[i] << " " << out2[i] << "\n";
    // }
    ASSERT_NEAR(out1[i], out2[i], 0.01);
  }
}

} // namespace jit
} // namespace torch
