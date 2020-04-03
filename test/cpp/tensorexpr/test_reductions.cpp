#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// Sum an array to a single value.
void testReduceSum1D() {
  KernelScope kernel_scope;

  Buffer b(BufHandle("b", {10}), kFloat);
  std::vector<float> in(10);
  for (int j = 0; j < 10; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor* c = Reduce("sum", {}, Sum(b), {{10, "m"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});
  ASSERT_EQ(out[0], 45);
}
// Sum a 2D tensor to a 1D tensor with dynamic shapes.
void testReduceSum2D() {
  KernelScope kernel_scope;

  const int M = 3;
  const int N = 7;

  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  Buffer b(BufHandle("b", {m, n}), kFloat);
  std::vector<float> in(M * N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      in[i * N + j] = j;
    }
  }

  std::vector<float> out(M, -1.f);

  Tensor* c = Reduce("sum", {{M, "m"}}, Sum(b), {{N, "n"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c, n, m});

  cg.call({in, out, 5, 7});

  float expected = 0;
  for (int i = 0; i < N; ++i) {
    expected += i;
  }

  for (int i = 0; i < M; ++i) {
    ASSERT_EQ(out[i], expected);
  }
}

// Sum a 3D tensor to both a 2D and 1D tensor, then reduce the 2D tensor flat to
// check our work.
void testReduceSum3D() {
  KernelScope kernel_scope;

  const int M = 10;
  VarHandle m("m", kInt);

  Buffer b(BufHandle("b", {2, 3, m}), kFloat);

  Tensor* c = Reduce("sum", {{2, "l"}, {3, "n"}}, Sum(b), {{m, "m"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c, m});

  std::vector<float> bData(2 * 3 * M, 0);
  std::vector<float> cData(2 * 3, 6.0f);
  std::vector<float> dData(2, 1.0f);
  std::vector<float> eData(2, 1.0f);

  for (int i = 0; i < 2 * 3; ++i) {
    for (int j = 0; j < M; ++j) {
      bData[i * M + j] = j;
    }
  }

  cg.call({bData, cData, M});
  float expected = 0;
  for (int i = 0; i < M; ++i) {
    expected += i;
  }

  for (int i = 0; i < 2 * 3; ++i) {
    ASSERT_EQ(cData[i], expected);
  }

  Tensor* d = Reduce("sum2", {{2, "l"}}, Sum(b), {{3, "n"}, {m, "m"}});
  LoopNest loop2({d});
  loop2.prepareForCodegen();
  Stmt* s2 = loop2.root_stmt();
  s2 = IRSimplifier::simplify(s2);

  SimpleIREvaluator cg2(s2, {b, d, m});
  cg2.call({bData, dData, M});

  // We're combining an additional dimension of 3, so the sum is 3x.
  expected = expected * 3;

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(dData[i], expected);
  }

  // This is the same as just reducing the original result across that axis.
  Buffer c_buf(BufHandle(c->func_var()), kFloat);
  Tensor* e = Reduce("sum3", {{2, "l"}}, Sum(c_buf), {{3, "m"}});
  LoopNest loop3({e});
  loop3.prepareForCodegen();
  Stmt* s3 = loop3.root_stmt();
  s3 = IRSimplifier::simplify(s3);

  SimpleIREvaluator cg3(s3, {c, e});
  cg3.call({cData, eData});

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(eData[i], expected);
  }
}

// Sum a large (10 D) Tensor 5 dimensions in.
void testReduceSum10D() {
  KernelScope kernel_scope;

  Buffer in_(BufHandle("in_", {2, 3, 2, 3, 2, 3, 2, 3, 2, 3}), kFloat);
  const int InputSize = 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3;
  Buffer out_(BufHandle("out_", {2, 3, 2, 3, 2}), kFloat);
  const int OutputSize = 2 * 3 * 2 * 3 * 2;

  std::vector<float> in(InputSize, 1.f);
  std::vector<float> out(OutputSize, -1.f);

  Tensor* c = Reduce(
      "sum",
      {{2, "a"}, {3, "b"}, {2, "c"}, {3, "d"}, {2, "e"}},
      Sum(in_),
      {{3, "f"}, {2, "g"}, {3, "h"}, {2, "i"}, {3, "j"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {in_, c});

  cg.call({in, out});

  float expected = InputSize / OutputSize;
  for (int i = 0; i < OutputSize; ++i) {
    ASSERT_EQ(out[i], expected);
  }
}

// Reduce via Mul rather than Add using a custom ReducePrototype.
void testReduceProduct() {
  KernelScope kernel_scope;

  const int M = 4;
  const int N = 4;

  Buffer b(BufHandle("b", {M, N}), kFloat);
  std::vector<float> in(M * N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      in[i * N + j] = 2 + j;
    }
  }

  std::vector<float> out(M, -1.f);

  ReducePrototype product(
      ExprHandle(1.f), [](ExprHandle a, ExprHandle b) { return a * b; }, b);

  Tensor* c = Reduce("product", {{M, "m"}}, product, {{N, "n"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});

  float expected = 1;
  for (int i = 0; i < N; ++i) {
    expected *= 2 + i;
  }

  for (int i = 0; i < M; ++i) {
    ASSERT_EQ(out[i], expected);
  }
}

// Foldl of a noncommutative operator (Sub) from a inital value.
void testReduceFoldSub() {
  KernelScope kernel_scope;

  Buffer b(BufHandle("b", {10}), kInt);
  std::vector<int> in(10);
  for (int j = 0; j < 10; ++j) {
    in[j] = j * 33;
  }

  std::vector<int> out(1, -1.f);
  ReducePrototype subtract(
      ExprHandle(1000), [](ExprHandle a, ExprHandle b) { return a - b; }, b);

  Tensor* c = Reduce("sum", {}, subtract, {{10, "m"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});

  int expected = 1000;
  for (int j = 0; j < 10; ++j) {
    expected -= j * 33;
  }

  ASSERT_EQ(out[0], expected);
}

// Maximum reductions.
void testReduceMax() {
  KernelScope kernel_scope;

  Buffer in_(BufHandle("b", {10}), kFloat);

  std::vector<float> in(10);
  std::vector<float> out(1, -1.f);
  for (int j = 0; j < 10; ++j) {
    in[j] = j;
  }

  Tensor* dm1 = Reduce("max", {}, Maximum(in_), {{10, "m"}});

  LoopNest loop({dm1});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);
  SimpleIREvaluator cg(s, {in_, dm1});

  cg.call({in, out});

  ASSERT_EQ(out[0], 9);

  Buffer in2_(BufHandle("b", {2, 5}), kFloat);
  std::vector<float> out2(2, -1.f);

  Tensor* m2d = Reduce("max", {{2, "n"}}, Maximum(in2_), {{5, "m"}});

  loop = LoopNest({m2d});
  loop.prepareForCodegen();
  s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg2(s, {in2_, m2d});
  cg2.call({in, out2});

  ASSERT_EQ(out2[0], 4);
  ASSERT_EQ(out2[1], 9);
}

// Minimum reduction, with custom initialization.
void testReduceMinCustomInitializer() {
  KernelScope kernel_scope;

  VarHandle minInit("minInit", kFloat);
  Buffer in_(BufHandle("b", {10}), kFloat);

  std::vector<float> in(10);
  std::vector<float> out(1, -1.f);
  for (int j = 0; j < 10; ++j) {
    in[j] = 10 + j;
  }

  Tensor* min = Reduce(
      "min",
      {},
      Minimum(
          [&](ParameterList& v) { return in_.call(v); }, ExprHandle(minInit)),
      {{10, "m"}});

  LoopNest loop({min});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {in_, min, minInit});

  // Works normally (note that out data starts lower than the correct
  // minimum).
  cg.call({in, out, std::numeric_limits<float>::max()});
  ASSERT_EQ(out[0], 10);

  // With an initalizer lower than the min, that's the min.
  cg.call({in, out, 5.f});
  ASSERT_EQ(out[0], 5);
}

// Example implementation of Any/All.
// TODO: this is very awkward without logical And/Or operators.
void testReduceAnyAll() {
  KernelScope kernel_scope;

  VarHandle searchValue("searchValue", kInt);
  Buffer b(BufHandle("b", {4, 10}), kInt);

  ReducePrototype anyEqSV(
      ExprHandle(0),
      [](ExprHandle a, ExprHandle b) {
        return CompareSelect::make(a, 1, 1, b, kEQ);
      },
      [&](ParameterList& v) {
        return CompareSelect::make(b.call(v), searchValue, kEQ);
      });

  Tensor* any = Reduce("anyEqual", {{4, "i"}}, anyEqSV, {{10, "j"}});

  LoopNest loop({any});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, any, searchValue});

  std::vector<int> in(40, 0);
  std::vector<int> out(4, 0);

  // input has 0-39 in 4 rows.
  for (int i = 0; i < 40; ++i) {
    in[i] = i;
  }
  cg.call({in, out, 1});

  // only the first row has 1
  ASSERT_EQ(out[0], 1);
  ASSERT_EQ(out[1], 0);
  ASSERT_EQ(out[2], 0);
  ASSERT_EQ(out[3], 0);

  cg.call({in, out, 15});

  // 15 in the 3rd row
  ASSERT_EQ(out[0], 0);
  ASSERT_EQ(out[1], 1);
  ASSERT_EQ(out[2], 0);
  ASSERT_EQ(out[3], 0);

  ReducePrototype allGTSV(
      ExprHandle(1),
      [](ExprHandle a, ExprHandle b) {
        return CompareSelect::make(a, 0, 0, b, kEQ);
      },
      [&](ParameterList& v) {
        return CompareSelect::make(b.call(v), searchValue, kGT);
      });

  Tensor* allGreaterThan =
      Reduce("allGreaterThan", {{4, "i"}}, allGTSV, {{10, "j"}});

  loop = LoopNest({allGreaterThan});
  loop.prepareForCodegen();
  s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg2(s, {b, allGreaterThan, searchValue});

  cg2.call({in, out, 11});

  // 11 is in row 2.
  ASSERT_EQ(out[0], 0);
  ASSERT_EQ(out[1], 0);
  ASSERT_EQ(out[2], 1);
  ASSERT_EQ(out[3], 1);

  cg2.call({in, out, -3});

  // All are positive.
  ASSERT_EQ(out[0], 1);
  ASSERT_EQ(out[1], 1);
  ASSERT_EQ(out[2], 1);
  ASSERT_EQ(out[3], 1);
}

void testReduceMatmul2D() {
  KernelScope kernel_scope;

  Buffer tA(BufHandle("tA", {3, 2}), kFloat);
  Buffer tB(BufHandle("tB", {2, 3}), kFloat);

  std::vector<float> tA_(6);
  std::vector<float> tB_(6);

  std::vector<float> out(9, -1.f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      tA_[i * 2 + j] = i * 2 + j;
      tB_[j * 3 + i] = i * 2 + j;
    }
  }

  Sum matmul([&](ParameterList& v) {
    ExprHandle m = v[0];
    ExprHandle n = v[1];
    ExprHandle k = v[2];
    return tA(m, k) * tB(k, n);
  });

  Tensor* mm = Reduce("mm", {{3, "m"}, {3, "n"}}, matmul, {{2, "k"}});

  LoopNest loop({mm});
  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {tA, tB, mm});
  cg.call({tA_, tB_, out});

  std::vector<float> expected(
      {1.f, 3.f, 5.f, 3.f, 13.f, 23.f, 5.f, 23.f, 41.f});

  for (int i = 0; i < 9; ++i) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

} // namespace jit
} // namespace torch
