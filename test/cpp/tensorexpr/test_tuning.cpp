#include "test/cpp/tensorexpr/test_base.h"
#include "test/cpp/tensorexpr/test_utils.h"

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/auto_tuner.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

#if USE_GPU
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#else
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#endif

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
      l.vectorize(split1);

      if (tail1) {
        For* outer2;
        For* split2;
        For* tail2;
        l.splitWithTail(tail1, 4, &outer2, &split2, &tail2);
        l.vectorize(split2);
      }
    }
  }
}

void initialRunHelper(
    LoopNest& loop,
    const std::vector<CodeGen::BufferArg>& symbols,
    const std::vector<CodeGen::CallArg>& args) {
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  LLVMCodeGen cg1(s, symbols);

  std::cout << "warming up\n";
  std::chrono::microseconds total(0);
  for (int i = 0; i < 20; ++i) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto start = std::chrono::high_resolution_clock::now();
    cg1.call(args);
    auto end = std::chrono::high_resolution_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }

  std::cout << "Pre tune ran in: " << (total / 20).count() << " us\n";
}

template <typename OutputType>
void goldenModelRunHelper(
    LoopNest& loop,
    std::vector<Tensor*> internal_tensors,
    const std::vector<CodeGen::BufferArg>& symbols,
    const std::vector<CodeGen::CallArg>& args,
    const std::vector<OutputType>& expected) {
  doKernelHeuristicOpts(internal_tensors, loop);
  Stmt* s3 = IRSimplifier::simplify(loop.root_stmt());
  LLVMCodeGen cg3(s3, symbols);
  std::chrono::microseconds total(0);

  for (int i = 0; i < 20; ++i) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto start = std::chrono::high_resolution_clock::now();
    cg3.call(args);
    auto end = std::chrono::high_resolution_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    for (int i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(expected[i], ((OutputType*)args.back().data())[i], 0.01);
    }
  }
  std::cout << "Golden ref ran in: " << (total / 20).count() << " us\n";
}

template <typename OutputType>
void tuningRunHelper(
    LoopNest& loop,
    const std::vector<CodeGen::BufferArg>& symbols,
    const std::vector<CodeGen::CallArg>& args,
    const std::vector<OutputType>& expected) {
  std::cout << *loop.root_stmt() << "\n";
  AutoTuner tuner(loop, symbols);
  tuner.run(20);

  LoopNest tunedLoop = tuner.getBestCandidate();
  tunedLoop.prepareForCodegen();
  Stmt* s2 = tunedLoop.root_stmt();
  s2 = IRSimplifier::simplify(s2);

  LLVMCodeGen cg2(s2, symbols);

  std::chrono::microseconds total(0);

  for (int i = 0; i < 20; ++i) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto start = std::chrono::high_resolution_clock::now();
    cg2.call(args);
    auto end = std::chrono::high_resolution_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    for (int i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(expected[i], ((OutputType*)args.back().data())[i], 0.01);
    }
  }

  std::cout << "Post tune ran in: " << (total / 20).count() << " us\n";
}

void testAutotuning1() {
  srand(time(NULL));
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

  /*
  for (int i0 = 0; i0 < 2048; i0++) {
    input1[i0] = t0[0 + i0 * 1];
  }
  for (int i0_1 = 0; i0_1 < 64; i0_1++) {
    for (int i1 = 0; i1 < 2048; i1++) {
      input2[i0_1, i1] = t1[(0 + i0_1 * 2048) + i1 * 1];
    }
  }
  for (int i0_2 = 0; i0_2 < 64; i0_2++) {
    for (int i1_1 = 0; i1_1 < 2048; i1_1++) {
      input3[i0_2, i1_1] = t2[(0 + i0_2 * 2048) + i1_1 * 1];
    }
  }
  for (int v = 0; v < 64; v++) {
    for (int v_1 = 0; v_1 < 2048; v_1++) {
      aten_add[v, v_1] = (input2(v, v_1)) + float(1) * (input3(v, v_1));
    }
  }
  for (int v_2 = 0; v_2 < 64; v_2++) {
    for (int v_3 = 0; v_3 < 2048; v_3++) {
      aten_add_1[v_2, v_3] = (aten_add(v_2, v_3)) + float(1) * (input1(v_3));
    }
  }
  for (int i0_3 = 0; i0_3 < 64; i0_3++) {
    for (int i1_2 = 0; i1_2 < 512; i1_2++) {
      prim_constantchunk[i0_3, i1_2] = aten_add(i0_3, i1_2 + 512);
    }
  }
  for (int i0_4 = 0; i0_4 < 64; i0_4++) {
    for (int i1_3 = 0; i1_3 < 512; i1_3++) {
      prim_constantchunk_1[i0_4, i1_3] = aten_add(i0_4, i1_3 + 1024);
    }
  }
  for (int i0_5 = 0; i0_5 < 64; i0_5++) {
    for (int i1_4 = 0; i1_4 < 512; i1_4++) {
      prim_constantchunk_2[i0_5, i1_4] = aten_add(i0_5, i1_4 + 1536);
    }
  }
  for (int i0_6 = 0; i0_6 < 64; i0_6++) {
    for (int i1_5 = 0; i1_5 < 512; i1_5++) {
      prim_constantchunk_3[i0_6, i1_5] = aten_add(i0_6, i1_5 + 0);
    }
  }
  for (int v_4 = 0; v_4 < 64; v_4++) {
    for (int v_5 = 0; v_5 < 512; v_5++) {
      aten_sigmoid[v_4, v_5] = 1.f / (1.f + (exp(-0.f - (prim_constantchunk(v_4,
v_5)))));
    }
  }
}*/

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

  std::vector<CodeGen::BufferArg> symbols = {tX, tA, tB, tY, mx};

  LoopNest initial({mx});
  initialRunHelper(initial, {tX, tA, tB, tY, mx}, {tX_, tA_, tB_, tY_, out1});

  LoopNest golden({mx});
  goldenModelRunHelper(
      golden, {ma}, {tX, tA, tB, tY, mx}, {tX_, tA_, tB_, tY_, out2}, out1);

  LoopNest initial2({mx});
  tuningRunHelper(
      initial2, {tX, tA, tB, tY, mx}, {tX_, tA_, tB_, tY_, out3}, out1);
}

/*
for (int i0 = 0; i0 < 2048; i0++) {
  input1[i0] = t0[0 + i0 * 1];
}
for (int i0_1 = 0; i0_1 < 64; i0_1++) {
  for (int i1 = 0; i1 < 2048; i1++) {
    input2[i0_1, i1] = t1[(0 + i0_1 * 2048) + i1 * 1];
  }
}
for (int i0_2 = 0; i0_2 < 64; i0_2++) {
  for (int i1_1 = 0; i1_1 < 2048; i1_1++) {
    input3[i0_2, i1_1] = t2[(0 + i0_2 * 2048) + i1_1 * 1];
  }
}
for (int v = 0; v < 64; v++) {
  for (int v_1 = 0; v_1 < 2048; v_1++) {
    aten_add[v, v_1] = (input2(v, v_1)) + float(1) * (input3(v, v_1));
  }
}
for (int v_2 = 0; v_2 < 64; v_2++) {
  for (int v_3 = 0; v_3 < 2048; v_3++) {
    aten_add_1[v_2, v_3] = (aten_add(v_2, v_3)) + float(1) * (input1(v_3));
  }
}
for (int i0_3 = 0; i0_3 < 64; i0_3++) {
  for (int i1_2 = 0; i1_2 < 512; i1_2++) {
    prim_constantchunk[i0_3, i1_2] = aten_add(i0_3, i1_2 + 512);
  }
}
for (int i0_4 = 0; i0_4 < 64; i0_4++) {
  for (int i1_3 = 0; i1_3 < 512; i1_3++) {
    prim_constantchunk_1[i0_4, i1_3] = aten_add(i0_4, i1_3 + 1024);
  }
}
for (int i0_5 = 0; i0_5 < 64; i0_5++) {
  for (int i1_4 = 0; i1_4 < 512; i1_4++) {
    prim_constantchunk_2[i0_5, i1_4] = aten_add(i0_5, i1_4 + 1536);
  }
}
for (int i0_6 = 0; i0_6 < 64; i0_6++) {
  for (int i1_5 = 0; i1_5 < 512; i1_5++) {
    prim_constantchunk_3[i0_6, i1_5] = aten_add(i0_6, i1_5 + 0);
  }
}
for (int v_4 = 0; v_4 < 64; v_4++) {
  for (int v_5 = 0; v_5 < 512; v_5++) {
    aten_sigmoid[v_4, v_5] = 1.f / (1.f + (exp(-0.f - (prim_constantchunk(v_4,
v_5)))));
  }
}
}*/

void testAutotuning2() {
  srand(time(NULL));
  KernelScope kernel_scope;
  int M = 256;
  int N = 2048;

  Buffer t0(BufHandle("tX", {N}, kFloat));
  Buffer t1(BufHandle("tA", {M, N}, kFloat));
  Buffer t2(BufHandle("tB", {M, N}, kFloat));

  Tensor* input1 = Compute(
      "input1", {{N, "i0"}}, [&](const ExprHandle& i0) { return t0(i0); });

  Tensor* input2 = Compute(
      "input2",
      {{M, "i0"}, {N, "i1"}},
      [&](const ExprHandle& i0, const ExprHandle& i1) { return t1(i0, i1); });

  Tensor* input3 = Compute(
      "input3",
      {{M, "i0"}, {N, "i1"}},
      [&](const ExprHandle& i0, const ExprHandle& i1) { return t2(i0, i1); });

  Tensor* add_1 = Compute(
      "aten_add1",
      {{M, "v"}, {N, "v"}},
      [&](const ExprHandle& v0, const ExprHandle& v1) {
        return input2->call(v0, v1) + input3->call(v0, v1);
      });

  Tensor* add_2 = Compute(
      "aten_add2",
      {{M, "v"}, {N, "v"}},
      [&](const ExprHandle& v0, const ExprHandle& v1) {
        return add_1->call(v0, v1) + input1->call(v1);
      });

  Tensor* chunk_0 = Compute(
      "chunk1",
      {{M, "i"}, {N / 4, "j"}},
      [&](const ExprHandle& i, const ExprHandle& j) {
        return add_2->call(i, j + ExprHandle(N / 4));
      });

  Tensor* chunk_1 = Compute(
      "chunk2",
      {{M, "i"}, {N / 4, "j"}},
      [&](const ExprHandle& i, const ExprHandle& j) {
        return add_2->call(i, j + ExprHandle(2 * (N / 4)));
      });

  Tensor* chunk_2 = Compute(
      "chunk3",
      {{M, "i"}, {N / 4, "j"}},
      [&](const ExprHandle& i, const ExprHandle& j) {
        return add_2->call(i, j + ExprHandle(3 * (N / 4)));
      });

  Tensor* chunk_3 = Compute(
      "chunk4",
      {{M, "i"}, {N / 4, "j"}},
      [&](const ExprHandle& i, const ExprHandle& j) {
        return add_2->call(i, j);
      });

  Tensor* sigmoid = Compute(
      "sigmoid",
      {{M, "i"}, {N / 4, "j"}},
      [&](const ExprHandle& i, const ExprHandle& j) {
        return ExprHandle(1.f) /
            (ExprHandle(1.f) +
             (Intrinsics::make(
                 kExp, ExprHandle(-0.f) - (chunk_3->call(i, j)))));
      });

  /*
for (int v_4 = 0; v_4 < 64; v_4++) {
for (int v_5 = 0; v_5 < 512; v_5++) {
aten_sigmoid[v_4, v_5] = 1.f / (1.f + (exp(-0.f - (prim_constantchunk(v_4,
v_5)))));*/

  std::vector<float> t0_(N, 0);
  std::vector<float> t1_(M * N, 0);
  std::vector<float> t2_(N * N, 0);

  std::vector<float> chunk1(M * N / 4, -1);
  std::vector<float> chunk2(M * N / 4, -1);
  std::vector<float> chunk3(M * N / 4, -1);
  std::vector<float> out1(M * N / 4, -1);
  std::vector<float> out2(M * N / 4, -1);
  std::vector<float> out3(M * N / 4, -1);

  for (int n = 0; n < N; ++n) {
    t0_[n] = n;
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      t1_[m * N + n] = m * N + n;
      t2_[m * N + n] = n * M + m;
    }
  }

  std::vector<CodeGen::BufferArg> symbols = {
      t0, t1, t2, chunk_0, chunk_1, chunk_2, sigmoid};

  LoopNest initial({chunk_0, chunk_1, chunk_2, sigmoid});
  initialRunHelper(
      initial, symbols, {t0_, t1_, t2_, chunk1, chunk2, chunk3, out1});

  LoopNest golden({chunk_0, chunk_1, chunk_2, sigmoid});
  goldenModelRunHelper(
      golden,
      {input1, input2, input3, add_1, add_2, chunk_3},
      symbols,
      {t0_, t1_, t2_, chunk1, chunk2, chunk3, out2},
      out1);

  LoopNest initial2({chunk_0, chunk_1, chunk_2, sigmoid});
  tuningRunHelper(
      initial2, symbols, {t0_, t1_, t2_, chunk1, chunk2, chunk3, out3}, out1);
  std::cout << "Done\n";
}

} // namespace jit
} // namespace torch
