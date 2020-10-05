#include <test/cpp/tensorexpr/test_base.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

static void verifyConstBounds(
    const TensorAccessBoundsInfo& access_info,
    const std::vector<std::pair<int, int>>& ref) {
  size_t ndim = ref.size();
  ASSERT_EQ(access_info.start.size(), ndim);
  ASSERT_EQ(access_info.stop.size(), ndim);
  for (size_t i = 0; i < ndim; i++) {
    if (ref[i].first >= 0) { // Negative values are used to skip the check
      ASSERT_TRUE(access_info.start[i]->isConstant());
      int start_i = immediateAs<int>(access_info.start[i]);
      ASSERT_EQ(start_i, ref[i].first);
    }
    if (ref[i].second >= 0) {
      ASSERT_TRUE(access_info.stop[i]->isConstant());
      int stop_i = immediateAs<int>(access_info.stop[i]);
      ASSERT_EQ(stop_i, ref[i].second);
    }
  }
}

void testBoundsInference_1() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 99}}
  KernelScope kernel_scope;
  ExprHandle n(100);
  Placeholder a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.data())[0], {{0, 99}});

  ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 99}});
}

void testBoundsInference_2() {
  // Verify that bounds inference works for the following example:
  // for i in 0..n:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, n-1}, {a, kLoad, 0, n-1}}
  KernelScope kernel_scope;
  VarHandle n("n", kInt);
  Placeholder a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.data())[0], {{0, -1}});

  ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b->buf())[0], {{0, -1}});
}

void testBoundsInference_3() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i] * a[i+10]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 109}}
  KernelScope kernel_scope;
  ExprHandle n(100);
  Placeholder a(BufHandle("a", {n + 10}, kFloat));
  Tensor* b = Compute("b", {{n, "i"}}, [&](const VarHandle& i) {
    return a.load(i) * a.load(i + 10);
  });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.data())[0], {{0, 109}});

  ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 99}});
}

void testBoundsInference_4() {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..200:
  //   for x in 0..320:
  //     c[y,x] = a[y,x] * b[y,x]
  KernelScope kernel_scope;
  ExprHandle W(320);
  ExprHandle H(200);
  Placeholder a(BufHandle("a", {H, W}, kFloat));
  Tensor* b = Compute(
      "b", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return x * y;
      });
  Tensor* c = Compute(
      "c", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return a.load(y, x) * b->call(y, x);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  Stmt* body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 199}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop body's scope
    auto bounds_info = inferBounds(body);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {-1, -1}});
  }
}

void testBoundsInference_5() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i]
  //
  // ==> split ==>
  //
  // for i_outer in 0..100/16:
  //   for i_inner in 0..16:
  //     b[i_outer * 16 + i_inner] = a[i_outer * 16 + i_inner]
  // for i_tail in 0..100%16:
  //   b[i_tail + (100/16)*16] = a[i_tail + (100/16)*16];
  KernelScope kernel_scope;
  ExprHandle n(100);
  Placeholder a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});

  For* outer;
  For* inner;
  For* tail;
  std::vector<For*> loops = l.getLoopStmtsFor(b);
  l.splitWithTail(loops[0], 16, &outer, &inner, &tail);

  {
    // Verify inferred bounds for the outer loop
    auto bounds_info = inferBounds(outer);
    ASSERT_EQ(bounds_info.size(), 2);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 95}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 95}});
  }
  {
    // Verify inferred bounds for the tail loop
    auto bounds_info = inferBounds(tail);
    ASSERT_EQ(bounds_info.size(), 2);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{96, 99}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{96, 99}});
  }
}

void testBoundsInference_6() {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..20:
  //   for x in 0..32:
  //     c[y,x] = a[y+100,x+100] * b[y*2,x*5]
  KernelScope kernel_scope;
  ExprHandle W(320);
  ExprHandle H(200);
  ExprHandle CW(32);
  ExprHandle CH(20);
  Placeholder a(BufHandle("a", {H, W}, kFloat));
  Tensor* b = Compute(
      "b", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return x * y;
      });
  Tensor* c = Compute(
      "c", {{CH, "y"}, {CW, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return a.load(y + 100, x + 100) * b->call(y * 2, x * 5);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  Stmt* body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{100, 119}, {100, 131}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 38}, {0, 155}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 19}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {100, 131}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {0, 155}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop body's scope
    auto bounds_info = inferBounds(body);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {-1, -1}});
  }
}

void testBoundsInferenceNonOverlapping() {
  KernelScope kernel_scope;
  ExprHandle H(3);
  Placeholder a(BufHandle("a", {10}, kFloat));
  Tensor* b =
      Compute("b", {{H, "x"}}, [&](const VarHandle& x) { return a.load(x); });
  Tensor* c = Compute(
      "c", {{H, "x"}}, [&](const VarHandle& x) { return a.load(x + H + 1); });
  LoopNest l({b, c});
  std::vector<For*> loops = NodeFinder<For>::find(l.root_stmt());

  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0:2], writes to b[0:2]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 2}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 2}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0+4:2+4], writes to c[0:2]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{4, 6}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 2}});
  }
  {
    // Infer bounds on the high level program.
    auto bounds_info = inferBounds(l.root_stmt());
    ASSERT_EQ(bounds_info.size(), 3);

    // Should be union of above 2 bounds.
    ASSERT_EQ(bounds_info.at(a.data()).size(), 2);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 2}});
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[1], {{4, 6}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 2}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 2}});
  }
}

void testBoundsInferenceAdjacent() {
  KernelScope kernel_scope;
  ExprHandle H(6);
  Placeholder a(BufHandle("a", {20}, kFloat));
  Tensor* b =
      Compute("b", {{H, "x"}}, [&](const VarHandle& x) { return a.load(x); });
  Tensor* c = Compute(
      "c", {{H, "x"}}, [&](const VarHandle& x) { return a.load(x + H); });
  LoopNest l({b, c});
  std::vector<For*> loops = NodeFinder<For>::find(l.root_stmt());

  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0:5], writes to b[0:5]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 5}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 5}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0+6:5+6], writes to c[0:5]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{6, 11}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 5}});
  }
  {
    // Infer bounds on the high level program.
    auto bounds_info = inferBounds(l.root_stmt());
    ASSERT_EQ(bounds_info.size(), 3);

    // Should be union of above 2 bounds, but this time the bounds of A can be
    // merged.
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 11}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 5}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 5}});
  }
}

void testMergeInferredBounds() {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("a", {10}, kFloat));

  // There are seven cases to consider in mergeTensorAccesses(A, B)
  //   * A is lower than B and does not overlap.
  //   * A is higher than B and does not overlap.
  //   * A overlaps B on both ends.
  //   * B overlaps A on both ends.
  //   * A overlaps B on the lower end. (equiv to B overlaps A on upper end).
  //   * A overlaps B on the upper end. (likewise covers reverse)
  //   * A and B are the same range.

  BoundsInfo info;
  // Test no overlap, both ways.
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(3)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(7)}});
  info[a.data()].push_back({kLoad, {new IntImm(9)}, {new IntImm(9)}});
  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[a.data()].size(), 3);

  ASSERT_EQ(res.at(a.data())[0].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[1].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[2].kind, kLoad);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{5, 7}});
  verifyConstBounds(res.at(a.data())[2], {{9, 9}});

  // Test full overlap, A over B.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(7)}});
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});

  // B over A.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(7)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});

  // Test partial overlap on the low end, A over B.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(7)}});
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{3, 7}});

  // Test partial overlap on the high end.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(2)}, {new IntImm(5)}});
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{2, 6}});

  // Test equality is deduped.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{4, 6}});
}

void testMergeInferredLoadStoreDiff() {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("a", {10}, kFloat));

  // Loads and Stores do not merge:
  BoundsInfo info;
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(7)}});
  info[a.data()].push_back({kStore, {new IntImm(3)}, {new IntImm(9)}});

  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[a.data()].size(), 2);
  ASSERT_EQ(res.at(a.data())[0].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[1].kind, kStore);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});
  verifyConstBounds(res.at(a.data())[1], {{3, 9}});

  // Do merge around the other kind of access:
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(3)}});
  info[a.data()].push_back({kStore, {new IntImm(3)}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(5)}});
  info[a.data()].push_back({kStore, {new IntImm(4)}, {new IntImm(8)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(7)}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});
  verifyConstBounds(res.at(a.data())[1], {{3, 8}});
}

void testMergeInferred2DBounds() {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("a", {10, 10}, kFloat));

  // Non overlapping in both dimensions:
  BoundsInfo info;
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(3), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(5), new IntImm(5)}, {new IntImm(9), new IntImm(9)}});

  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[a.data()].size(), 2);
  ASSERT_EQ(res.at(a.data())[0].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[1].kind, kLoad);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}, {1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{5, 9}, {5, 9}});

  // Overlapping in a single dimension should mean we cannot merge.
  // First dimension:
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(3), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(2), new IntImm(5)}, {new IntImm(9), new IntImm(9)}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}, {1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{2, 9}, {5, 9}});

  // Second dimension:
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(3), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(5), new IntImm(2)}, {new IntImm(9), new IntImm(9)}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}, {1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{5, 9}, {2, 9}});

  // Overlapping in both dimensions:
  // {1-6, 1-3) | {4-9, 2,7} => {1,9, 1,7}
  // TODO: this will overestimate and we should fix it.
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(6), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(4), new IntImm(2)}, {new IntImm(9), new IntImm(7)}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 9}, {1, 7}});
}

void testMergeAdjacentBounds() {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("a", {10}, kFloat));

  // Adjacent but not overlapping bounds can be merged.
  // e.g. {1-4} | {5-9} => {1-9}
  BoundsInfo info;
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(9)}});
  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 9}});

  // And on the other side:
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(9)}});
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(4)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 9}});

  // One space gap is enough to prevent merging:
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {new IntImm(9)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 4}});
  verifyConstBounds(res.at(a.data())[1], {{6, 9}});
}

std::pair<std::string, std::string> boundAsStringPair(
    TensorAccessBoundsInfo& info,
    size_t idx = 0) {
  std::ostringstream start, stop;
  start << *info.start[idx];
  stop << *info.stop[idx];
  return {start.str(), stop.str()};
}

void testMergeSymbolicBounds() {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("a", {10}, kFloat));
  VarHandle W("W", kInt);
  VarHandle X("X", kInt);
  VarHandle Y("Y", kInt);
  VarHandle Z("Z", kInt);

  // Can do nothing with fully symbolic bounds:
  BoundsInfo info;
  info[a.data()].push_back({kLoad, {W.node()}, {Z.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // Can merge if the difference between bounds is constant and enclosing.
  // {X-Y} | {X-5 - Y+10} => {X-5 - Y+10}
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back({kLoad,
                            {new Sub(X.node(), new IntImm(5))},
                            {new Add(Y.node(), new IntImm(10))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);

  // Cannot merge otherwise.
  // {X-Y} | {X+5 - Y+10} => could be 2 groups if Y < X+5.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back({kLoad,
                            {new Add(X.node(), new IntImm(5))},
                            {new Add(Y.node(), new IntImm(10))}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // Can't merge if there's a gap of at least one element:
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // Can't even though the high of the first bound is above the low of the
  // second, X can == 6 and Y can == 4 so this can't merge in all cases.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // If either side is equal, they must be overlapping.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Z.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  auto pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Max(Y, Z, 1)");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back({kLoad, {Z.node()}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "Min(X, Z, 1)");
  ASSERT_EQ(pair.second, "Y");

  // If either side is only one apart, they must be adjacent.
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new Add(X.node(), new IntImm(1))}, {Z.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Max(Y, Z, 1)");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back(
      {kLoad, {Z.node()}, {new Sub(Y.node(), new IntImm(1))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "Min(X, Z, 1)");
  ASSERT_EQ(pair.second, "Y");

  // If either side is 2 apart, they may not be overlapping.
  // in this case if Y == X+1 they don't overlap.
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new Add(X.node(), new IntImm(2))}, {Z.node()}});
  info[a.data()].push_back(
      {kLoad, {X.node()}, {new Sub(Y.node(), new IntImm(1))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // In this case they may not overlap if X == Y.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back(
      {kLoad, {Z.node()}, {new Sub(Y.node(), new IntImm(2))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
}

void testMergeSymbolicAdjacent() {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("a", {10}, kFloat));
  VarHandle X("X", kInt);
  VarHandle Y("Y", kInt);

  BoundsInfo info;
  // Can merge if a range is adjacent:
  // {X-5} | {6-Y} => {X-Y}
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(5)}});
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {Y.node()}});
  BoundsInfo res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  auto pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {Y.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(5)}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {Y.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {Y.node()}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  // If either the lower or upper bound is adjacent the range then they must
  // overlap, even if we don't know the extent.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {X.node()}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "5");
  ASSERT_EQ(pair.second, "Max(X, Y, 1)");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {Y.node()}, {new IntImm(5)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "Min(X, Y, 1)");
  ASSERT_EQ(pair.second, "6");
}

void testBoundNormalize() {
  KernelScope kernel_scope;

  using namespace mem_dependency;

  auto CB = [](ExprHandle s, ExprHandle e, ExprHandle st) {
    return Bound(s.node(), e.node(), st.node());
  };

  auto EQ = [](Bound a, Bound b) -> bool {
    std::cout << "EQ ";
    a.print();
    std::cout << " ";
    b.print();
    std::cout << "\n";
    return exprEquals(a.start, b.start) && exprEquals(a.end, b.end) &&
        exprEquals(a.stride, b.stride);
  };

  // Stride 1 means unchanged.
  ASSERT_TRUE(EQ(normalizeBound(CB(0, 5, 1)), CB(0, 5, 1)));

  // constrict bound end to be on the stride.
  ASSERT_TRUE(EQ(normalizeBound(CB(0, 5, 2)), CB(0, 4, 2)));
  ASSERT_TRUE(EQ(normalizeBound(CB(1, 6, 2)), CB(1, 5, 2)));
  ASSERT_TRUE(EQ(normalizeBound(CB(1, 6, 7)), CB(1, 1, 7)));
  // don't do anything if already normalized.
  ASSERT_TRUE(EQ(normalizeBound(CB(0, 4, 2)), CB(0, 4, 2)));

  VarHandle x("x", kInt);
  // Don't do end normalization with dynamic bounds.
  ASSERT_TRUE(EQ(normalizeBound(CB(0, x, 3)), CB(0, x, 3)));
  ASSERT_TRUE(EQ(normalizeBound(CB(x, 10, 3)), CB(x, 10, 3)));

  // Can't do end normalization with dynamic strides.
  ASSERT_TRUE(EQ(normalizeBound(CB(0, 10, x)), CB(0, 10, x)));

  // Normalize negative strides.
  ASSERT_TRUE(EQ(normalizeBound(CB(10, 0, -1)), CB(0, 10, 1)));
  ASSERT_TRUE(EQ(normalizeBound(CB(10, 0, -2)), CB(0, 10, 2)));
  // Shift start index if necessary.
  ASSERT_TRUE(EQ(normalizeBound(CB(10, 0, -3)), CB(1, 10, 3)));

  // Negative strides with dynamic bounds.
  // This gets complex because the start of the bound determines the offset from
  // the stride. The new start of the bound will always be compound since it
  // must include the difference in start and end offset mod the stride.
  ASSERT_TRUE(EQ(
      normalizeBound(CB(100, x, -3)), CB((ExprHandle(1) - x) % 3 + x, 100, 3)));
  ASSERT_TRUE(EQ(normalizeBound(CB(x, 11, -3)), CB((x - 2) % 3 + 11, x, 3)));
}

void testBoundOverlap() {
  KernelScope kernel_scope;

  using namespace mem_dependency;

  auto CB = [](int s, int e, int st) {
    return Bound(new IntImm(s), new IntImm(e), new IntImm(st));
  };

  // Sanity check 3 overlap cases.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 0, 0), CB(0, 0, 0)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 3, 1), CB(2, 5, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 0, 0), CB(1, 1, 0)));

  // Partial overlap works in either order.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, 1), CB(7, 14, 1)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(7, 14, 1), CB(0, 10, 1)));

  // Total Overlap works when one bound encloses the other, and returns which.
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(2, 15, 1), CB(7, 9, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(2, 15, 1), CB(0, 16, 1)));

  // Total overlap works when the bounds are an identical range, returns
  // TotalOverlapB.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(2, 15, 1), CB(2, 15, 1)));

  // Total overlap when only one end of the bound matches.
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(2, 15, 1), CB(2, 10, 1)));
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(2, 15, 1), CB(3, 15, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(2, 10, 1), CB(2, 15, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(3, 15, 1), CB(2, 15, 1)));

  // No overlap when a < b.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 2, 1), CB(5, 10, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(2, 2, 1), CB(3, 3, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(100, 120, 1), CB(130, 130, 1)));

  // No overlap when a > b.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(5, 10, 1), CB(0, 2, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(3, 3, 1), CB(2, 2, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(130, 130, 1), CB(100, 120, 1)));

  // No overlap when adjacent.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 100, 1), CB(101, 120, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(2, 3, 1), CB(0, 1, 1)));

  // Partial overlap when middle bounds match.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 100, 1), CB(100, 120, 1)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 2, 1), CB(2, 4, 1)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(100, 120, 1), CB(0, 100, 1)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(2, 3, 1), CB(1, 2, 1)));

  // Total overlap when one bound is single length over one end of the other.
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(2, 15, 1), CB(15, 15, 1)));
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(2, 15, 1), CB(2, 2, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(2, 2, 1), CB(2, 15, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(15, 15, 1), CB(2, 15, 1)));
}

void testBoundOverlapStrides() {
  KernelScope kernel_scope;

  using namespace mem_dependency;

  auto CB = [](int s, int e, int st) {
    return Bound(new IntImm(s), new IntImm(e), new IntImm(st));
  };

  // Same bounds with same stride overlaps totally.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, 1), CB(0, 10, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, 2), CB(0, 10, 2)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, 10), CB(0, 10, 10)));

  // Same bounds with a different stride may overlap partially.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, 2), CB(0, 10, 3)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, 3), CB(0, 10, 2)));

  // But may overlap totally if one stride is a clean multiple of the other.
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(0, 10, 1), CB(0, 10, 2)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, 2), CB(0, 10, 1)));
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(0, 100, 7), CB(0, 100, 21)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 100, 33), CB(0, 100, 3)));

  // Don't overlap if the area covered by the strided bound is totally distinct.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 10, 2), CB(1, 10, 2)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 10, 3), CB(2, 10, 3)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 10, 2), CB(5, 10, 2)));

  // Don't overlap if the strides are not distinct but the range is also not
  // distinct.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 10, 3), CB(11, 15, 2)));

  // Partial overlap with same strides and partially overlapping regions.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, 2), CB(6, 14, 2)));

  // No overlap if the stride is greater than the difference in the region.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 10, 2), CB(7, 14, 2)));

  // Partial overlap with distinct strides and partially overlapping regions.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, 3), CB(7, 11, 2)));

  // Total overlap even if the bounds are not equal so long as the difference is
  // smaller than the stride.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, 2), CB(0, 11, 2)));
  // Even if the second bound is smaller, it may totally enclose the first bound
  // if the difference is smaller than the stride.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 11, 2), CB(0, 10, 2)));

  // smaller bound overlaps inside a strided region.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(2, 2, 0), CB(0, 20, 2)));
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(0, 20, 2), CB(2, 2, 0)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(2, 2, 0), CB(1, 19, 2)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(1, 19, 2), CB(2, 2, 0)));
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(0, 6, 3), CB(0, 0, 1)));

  // Range overlaps with mismatched strides.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 6, 3), CB(1, 3, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 6, 3), CB(1, 2, 1)));

  ASSERT_EQ(PartialOverlap, boundOverlap(CB(2, 12, 7), CB(6, 10, 3)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(2, 12, 7), CB(0, 8, 3)));

  // Checking edge cases, LCM(11, 17) is 187.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(11, 200, 11), CB(0, 200, 17)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(11, 187, 11), CB(0, 188, 17)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(11, 186, 11), CB(0, 188, 17)));
}

void testBoundOverlapNegativeStrides() {
  KernelScope kernel_scope;

  using namespace mem_dependency;

  auto CB = [](int s, int e, int st) {
    return Bound(new IntImm(s), new IntImm(e), new IntImm(st));
  };

  // Negative strides still work.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(10, 0, -1), CB(10, 0, -1)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(10, 3, -1), CB(5, 2, -1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(10, 6, -1), CB(5, 1, -1)));

  // Distinct strides work with negative bounds with different start offsets.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(10, 0, -2), CB(9, 0, -2)));

  // Total Overlap favours the side with the smaller (absolute) stride.
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(100, 0, -3), CB(100, 0, -6)));

  // Mixed sign of strides works.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(10, 0, -1), CB(0, 10, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, 1), CB(10, 0, -1)));

  // Distinct strides work with mixed signs.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(10, 0, -2), CB(1, 10, 2)));
}

void testBoundOverlapSymbolic() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  VarHandle w("w", kInt);

  using namespace mem_dependency;

  auto CB = [](ExprHandle s, ExprHandle e, ExprHandle st) {
    return Bound(s.node(), e.node(), st.node());
  };

  // Sanity check cases where the start and end is symbolic but the diff is
  // constant.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(x, x, 0), CB(x, x, 0)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, x + 3, 1), CB(x + 2, x + 5, 1)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(x, x, 0), CB(x + 1, x + 1, 0)));

  // Strides work as normal even if we don't know the exact bounds.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(x, y, 1), CB(x, y, 1)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(x, y, 2), CB(x, y, 2)));
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(x, y, 4), CB(x, y, 2)));
  ASSERT_EQ(TotalOverlapA, boundOverlap(CB(x, y, 2), CB(x, y, 4)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, y, 4), CB(x, y, 3)));

  // Negative strides with unknown bounds.
  // If the stride is 1, we can reverse the negative bound safely.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(x, y, 1), CB(y, x, -1)));
  // if the stride is > 1, when we reverse we may potentially need to offset x
  // with y%2 to factor in the case where x and y are on distinct strides of 2.
  // Since this is unknown we cant be sure of total overlap.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, y, 2), CB(y, x, -2)));

  // We can't infer the sign of y, so cannot tell whether adding y is larger or
  // smaller than y/2.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, x + y, 1), CB(x, x + y / 2, 1)));

  // No information about this bound, have to take the most conservative option:
  // there may be an overlap.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, y, 1), CB(z, w, 1)));

  // Math on opaque terms works.
  ASSERT_EQ(
      TotalOverlapB, boundOverlap(CB(x + w, y - z, 1), CB(x + w, y - z, 1)));
  // Even requiring simplification.
  ASSERT_EQ(
      TotalOverlapB, boundOverlap(CB(x - w - w, y, 1), CB(x - w * 2, y, 1)));

  // Symbolic strides.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, x), CB(0, 10, x)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 5, x), CB(5, 10, x)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, x), CB(0, 10, y)));

  // Can't determine if x * 2 > x because we don't know the sign of x.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, x), CB(0, 10, x * 2)));

  // Can't determine if x * y > x because we don't know if y is zero.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10, x), CB(0, 10, x * y)));
  // Can determine strides if they simplify to something constant.
  ASSERT_EQ(TotalOverlapB, boundOverlap(CB(0, 10, x), CB(0, 10, x * (y / y))));

  // Distinct strides with unknown bounds.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(x, x + 10, 2), CB(x + 1, x + 10, 2)));
  // Even when we don't know if end overlaps.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(x, x + y, 3), CB(x + 1, x + y, 3)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(x, x + y, 5), CB(x + 2, x + 10, 5)));
}

void testBoundSubtract() {
  KernelScope kernel_scope;

  using namespace mem_dependency;

  auto CB = [](int s, int e, int st) {
    return Bound(new IntImm(s), new IntImm(e), new IntImm(st));
  };

  auto EQ = [](std::vector<Bound> a, std::vector<Bound> b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (auto i = 0; i != a.size(); ++i) {
      if (!exprEquals(a[i].start, b[i].start) ||
          !exprEquals(a[i].end, b[i].end) ||
          !exprEquals(a[i].stride, b[i].stride)) {
        return false;
      }
    }
    return true;
  };

  // One element subtract.
  ASSERT_EQ(subtractBound(CB(0, 0, 0), CB(0, 0, 0)).size(), 0);
  ASSERT_EQ(subtractBound(CB(5, 5, 1), CB(5, 5, 1)).size(), 0);
  // Different strides.
  ASSERT_EQ(subtractBound(CB(5, 5, 1), CB(5, 5, 3)).size(), 0);

  // No Overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5, 3), CB(2, 2, 3)), {CB(5, 5, 3)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 0, 3), CB(2, 2, 1)), {CB(0, 0, 3)}));
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5, 3), CB(1, 2, 3)), {CB(5, 5, 3)}));

  // no stride, one side overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5, 1), CB(4, 7, 1)), {CB(1, 3, 1)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 5, 1), CB(5, 7, 1)), {CB(0, 4, 1)}));
  ASSERT_TRUE(EQ(subtractBound(CB(4, 5, 1), CB(1, 4, 1)), {CB(5, 5, 1)}));
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5, 1), CB(0, 4, 1)), {CB(5, 5, 1)}));

  // no stride, both sides overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5, 1), CB(0, 7, 1)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5, 1), CB(5, 7, 1)), {}));

  // no strides, internal overlap.
  ASSERT_TRUE(
      EQ(subtractBound(CB(1, 5, 1), CB(2, 3, 1)), {CB(1, 1, 1), CB(4, 5, 1)}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 5, 1), CB(2, 4, 1)), {CB(0, 1, 1), CB(5, 5, 1)}));

  // Strides match, low overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, 2), CB(4, 10, 2)), {CB(0, 2, 2)}));
  // Strides match, high overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(6, 10, 2), CB(2, 7, 2)), {CB(8, 10, 2)}));
  // Strides match, total overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(6, 10, 2), CB(2, 17, 2)), {}));
  // Strides match internal overlap.
  ASSERT_TRUE(EQ(
      subtractBound(CB(0, 10, 2), CB(2, 7, 2)), {CB(0, 0, 2), CB(8, 10, 2)}));
  // Strides match, no overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(6, 10, 2), CB(11, 17, 2)), {CB(6, 10, 2)}));

  // Strides equal but distinct offsets.
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, 2), CB(3, 10, 2)), {CB(0, 10, 2)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, 6), CB(3, 10, 6)), {CB(0, 6, 6)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, 3), CB(2, 10, 3)), {CB(0, 9, 3)}));

  // A has larger multiple stride - total overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, 4), CB(0, 10, 2)), {}));
  // B has a larger multiple stride.
  ASSERT_TRUE(EQ(subtractBound(CB(0, 100, 1), CB(0, 100, 2)), {CB(1, 99, 2)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, 2), CB(0, 10, 4)), {CB(2, 10, 4)}));

  // A has a larger stride but also larger bounds.
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 100, 4), CB(20, 80, 2)),
         {CB(0, 16, 4), CB(84, 100, 4)}));
  // A has a smaller stride, but also smaller bounds.
  ASSERT_TRUE(EQ(subtractBound(CB(20, 80, 2), CB(0, 100, 4)), {CB(22, 78, 4)}));
  // Sanity check the opposite.
  ASSERT_TRUE(EQ(subtractBound(CB(20, 80, 4), CB(0, 100, 2)), {}));
  // If a has a larger bound but smaller strides, we have a head and tail with
  // the small strides and a body with the large strides.
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 100, 2), CB(20, 80, 4)),
         {CB(0, 18, 2), CB(22, 78, 4), CB(82, 100, 2)}));

  // May need to slice the bound into multiple offsets of the larger B stride.
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 100, 4), CB(0, 100, 20)),
         {CB(4, 84, 20), CB(8, 88, 20), CB(12, 92, 20), CB(16, 96, 20)}));
  // If the strides do not divide evenly, we may create new bounds with stride =
  // LCM of input strides.
  ASSERT_TRUE(EQ(subtractBound(CB(0, 31, 3), CB(0, 31, 2)), {CB(3, 27, 6)}));
  // Note the ends of the bounds may not match.
  ASSERT_TRUE(EQ(
      subtractBound(CB(0, 31, 2), CB(0, 31, 3)), {CB(2, 26, 6), CB(4, 28, 6)}));

  // If a bound is offset from b, still works with different starts.
  ASSERT_TRUE(EQ(
      subtractBound(CB(4, 31, 2), CB(0, 31, 3)), {CB(4, 28, 6), CB(8, 26, 6)}));
  ASSERT_TRUE(EQ(
      subtractBound(CB(3, 31, 2), CB(0, 31, 3)), {CB(5, 29, 6), CB(7, 31, 6)}));

  // Result groups can get large, but there's a cut off at 20 groups.
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 25, 3), CB(0, 25, 7)),
         {CB(3, 24, 21),
          CB(6, 6, 21),
          CB(9, 9, 21),
          CB(12, 12, 21),
          CB(15, 15, 21),
          CB(18, 18, 21)}));

  // Offset strides don't affect the head and tail not included in the b range,
  // and so the results may have a mix of strides.
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 50, 2), CB(6, 31, 3)),
         {CB(0, 4, 2), CB(8, 26, 6), CB(10, 28, 6), CB(32, 50, 2)}));
}

void testBoundSubtractSymbolic() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  VarHandle w("w", kInt);

  using namespace mem_dependency;

  auto CB = [](ExprHandle s, ExprHandle e, ExprHandle st) {
    return Bound(s.node(), e.node(), st.node());
  };

  auto EQ = [](std::vector<Bound> a, std::vector<Bound> b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (auto i = 0; i != a.size(); ++i) {
      if (!exprEquals(a[i].start, b[i].start) ||
          !exprEquals(a[i].end, b[i].end) ||
          !exprEquals(a[i].stride, b[i].stride)) {
        return false;
      }
    }
    return true;
  };

  // One element subtract.
  ASSERT_TRUE(EQ(subtractBound(CB(x, x, 0), CB(x, x, 0)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x + 1, x + 1, 0), CB(x + 1, x + 1, 0)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x * 2, x * 2, 0), CB(x * 2, x * 2, 0)), {}));

  // Subtract constant range low.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10, 1), CB(x, x + 4, 1)),
         {CB(x + 5, x + 10, 1)}));
  // Subtract constant range high.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10, 2), CB(x + 6, x + 12, 2)),
         {CB(x, x + 4, 2)}));
  // Subtract constant range total overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(x, x + 10, 2), CB(x, x + 10, 2)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x + 2, x + 10, 1), CB(x, x + 12, 1)), {}));
  // Subtract constant range internal.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10, 1), CB(x + 3, x + 7, 1)),
         {CB(x, x + 2, 1), CB(x + 8, x + 10, 1)}));

  // distinct strides with unknown bounds.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10, 2), CB(x + 1, x + 10, 2)),
         {CB(x, x + 10, 2)}));
  // Even if we don't know the extent (but bound end will be normalized).
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10, 4), CB(x + 1, y, 4)), {CB(x, x + 8, 4)}));

  // A has larger multiple stride - total overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(x, x + 10, 4), CB(x, x + 10, 2)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, y, 4), CB(x, y, 2)), {}));
  // B has a larger multiple stride.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10, 1), CB(x, x + 10, 2)),
         {CB(x + 1, x + 9, 2)}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, y, 1), CB(x, y, 2)), {CB(x + 1, y, 2)}));

  // B totally closes with room to spare and has smaller strides.
  ASSERT_TRUE(EQ(subtractBound(CB(x + 4, x + 12, 4), CB(x, x + 20, 2)), {}));
  // If a has a larger bound but smaller strides, we have a head and tail with
  // the small strides and a body with the large strides.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 100, 2), CB(x + 20, x + 80, 4)),
         {CB(x, x + 18, 2), CB(x + 22, x + 78, 4), CB(x + 82, x + 100, 2)}));

  // Slicing into multiple bounds. We can't tell if y is a part of any
  // particular stride, so the end bound stays at y.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, y, 4), CB(x, y, 12)),
         {CB(x + 4, y, 12), CB(x + 8, y, 12)}));

  // If the strides do not divide evenly, we may create new bounds with stride =
  // LCM of input strides.
  ASSERT_TRUE(EQ(subtractBound(CB(0, x, 3), CB(0, x + 100, 2)), {CB(3, x, 6)}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, 100, 3), CB(x, 200, 2)), {CB(x + 3, 100, 6)}));

  // If a bound is offset from b, still works with different starts.
  ASSERT_TRUE(EQ(subtractBound(CB(1, x, 3), CB(0, x + 100, 2)), {CB(1, x, 6)}));
  ASSERT_TRUE(EQ(subtractBound(CB(2, x, 3), CB(0, x + 100, 2)), {CB(5, x, 6)}));

  ASSERT_TRUE(
      EQ(subtractBound(CB(1, x, 2), CB(0, x + 100, 3)),
         {CB(1, x, 6), CB(5, x, 6)}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(2, x, 2), CB(0, x + 100, 3)),
         {CB(2, x, 6), CB(4, x, 6)}));

  // Size is inferable but not constant, only works with a single var.
  ASSERT_TRUE(EQ(subtractBound(CB(0, x, 1), CB(0, x * 2, 1)), {}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, x * 2, 1), CB(0, x - 1, 1)), {CB(x, x * 2, 1)}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, x * 2, 7), CB(x, x * 2, 7)), {CB(0, x - 1, 7)}));

  // Size is not inferable.
  ASSERT_TRUE(EQ(subtractBound(CB(x, y, 1), CB(z, w, 1)), {CB(x, y, 1)}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, y, 1), CB(x, z, 1)), {CB(x, y, 1)}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, y, 1), CB(0, x, 1)), {CB(x, y, 1)}));
}

void testBoundSubtractSymbolicStrides() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  VarHandle w("w", kInt);

  using namespace mem_dependency;

  auto CB = [](ExprHandle s, ExprHandle e, ExprHandle st) {
    return Bound(s.node(), e.node(), st.node());
  };

  auto EQ = [](std::vector<Bound> a, std::vector<Bound> b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (auto i = 0; i != a.size(); ++i) {
      if (!exprEquals(a[i].start, b[i].start) ||
          !exprEquals(a[i].end, b[i].end) ||
          !exprEquals(a[i].stride, b[i].stride)) {
        return false;
      }
    }
    return true;
  };

  // One element subtract.
  ASSERT_TRUE(EQ(subtractBound(CB(x, x, x), CB(x, x, x)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, x, y), CB(x, x, y)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, x, y), CB(x, x, z)), {}));

  // Same strides - safe if the low end is untouched and there is no tail.
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5, x), CB(4, 7, x)), {CB(1, 3, x)}));
  ASSERT_TRUE(EQ(
      subtractBound(CB(x, x + 5, y), CB(x + 2, x + 10, y)), {CB(x, x + 1, y)}));
  // Not safe otherwise, default to returning A.
  ASSERT_TRUE(EQ(subtractBound(CB(4, 5, x), CB(1, 4, x)), {CB(4, 5, x)}));
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5, x), CB(2, 4, x)), {CB(1, 5, x)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, x), CB(2, 7, x)), {CB(0, 10, x)}));

  ASSERT_TRUE(
      EQ(subtractBound(CB(x + 3, x + 5, y), CB(x, x + 3, y)),
         {CB(x + 3, x + 5, y)}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 5, y), CB(x, x + 3, y)), {CB(x, x + 5, y)}));
  ASSERT_TRUE(EQ(
      subtractBound(CB(x, x + 5, y), CB(x + 1, x + 3, y)), {CB(x, x + 5, y)}));

  // Full overlap still works.
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5, x), CB(1, 5, x)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5, x), CB(0, 7, x)), {}));

  ASSERT_TRUE(EQ(subtractBound(CB(x, y, z), CB(x, y, z)), {}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(x + 1, x + 10, z), CB(x + 1, x + 10, z)), {}));

  ASSERT_TRUE(EQ(subtractBound(CB(x + 10, x + 90, z), CB(x, x + 100, z)), {}));

  // No overlap still works.
  ASSERT_TRUE(EQ(subtractBound(CB(6, 10, x), CB(11, 17, x)), {CB(6, 10, x)}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(x + 6, x + 10, z), CB(x + 11, x + 17, z)),
         {CB(x + 6, x + 10, z)}));

  // Stride is greater but clearly a multiple.
  ASSERT_TRUE(EQ(subtractBound(CB(0, 10, x * 2), CB(0, 10, x)), {}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 100, x), CB(0, 100, x * 2)), {CB(1, 100, x * 2)}));
  ASSERT_TRUE(
      EQ(subtractBound(CB(0, 10, x), CB(0, 10, x * 3)),
         {CB(1, 10, x * 3), CB(2, 10, x * 3)}));

  // If the strides differ then we're out of luck since we can't work out the
  // LCM.
  ASSERT_TRUE(EQ(subtractBound(CB(0, 100, x), CB(0, 100, y)), {CB(0, 100, x)}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, x, x), CB(y, y, y)), {CB(x, x, x)}));
}

void testMemDependencyCheckerSimple() {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  mem_dependency::RWDependencyAnalysis analyzer;

  /*
  Stmt* stmt = Block::make(
      {Store::make(a, {x}, Load::make(b, {x}, 1), 1),
       Store::make(a, {x}, Load::make(b, {x + 1}, 1), 1),
       Store::make(c, {x}, Load::make(a, {x}, 1), 1),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({
               Store::make(a, {x}, Add::make(Load::make(a, {x}, 1), 1), 1),
               Store::make(a, {0}, 3, 1),
               Store::make(a, {x}, Add::make(Load::make(a, {x}, 1), 1), 1),
           }),
           nullptr),
       Store::make(b, {x}, Load::make(a, {x}, 1), 1),
       Store::make(b, {x + 1}, Load::make(a, {x}, 1), 1),
       Store::make(a, {x}, Load::make(c, {x}, 1), 1)});
  */
  // Stmt* stmt = Block::make({Store::make(a, {x}, Load::make(b, {x}, 1), 1),
  //                           Store::make(a, {x + 1}, Load::make(a, {x}, 1),
  //                           1), Store::make(c, {0}, Load::make(a, {y}, 1),
  //                           1)});
  //
  Stmt* stmt = Block::make(
      // {Store::make(a, {0}, 2, 1),
      //  Store::make(a, {1}, Load::make(a, {0}, 1), 1),
      {For::make(x, 0, 10, Store::make(a, {x}, Load::make(a, {x}, 1), 1)),
       For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}, 1), 1)),
       For::make(x, 0, 9, Store::make(a, {x}, Load::make(a, {x + 1}, 1), 1)),
       For::make(
           x,
           0,
           9,
           Store::make(
               a,
               {ExprHandle(9) - x},
               Load::make(a, {ExprHandle(8) - x}, 1),
               1)),
       For::make(
           x,
           0,
           10,
           Store::make(a, {x}, Load::make(a, {ExprHandle(9) - x}, 1), 1))});
  // Store::make(b, {0}, Load::make(a, {4}, 1), 1),
  // Store::make(b, {1}, Load::make(a, {1}, 1), 1)});

  stmt->accept(&analyzer);

  std::cout << *stmt << "\n";
  for (auto& wi : analyzer.getHistory()) {
    wi->print();
  }
}

void testMemDependencyCheckerLoop() {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  BufHandle d("D", {5}, kInt);
  BufHandle e("E", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        y,
        0,
        10,
        Block::make(
            {Store::make(a, {y}, Add::make(Load::make(a, {y}, 1), 1), 1)}));

    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        y,
        0,
        10,
        Block::make({Store::make(
            a, {y + 1}, Add::make(Load::make(a, {y + 1}, 1), 1), 1)}));

    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Block::make(
            {Store::make(e, {0}, Add::make(Load::make(e, {0}, 1), x), 1)}));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Block::make(
            {Store::make(e, {0}, Add::make(Load::make(e, {0}, 1), x), 1)}));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Block::make(
            {Store::make(d, {y}, Add::make(Load::make(d, {y}, 1), x), 1)}));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt =
        For::make(x, 0, 10, Store::make(b, {x}, Load::make(b, {x + 1}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(c, {x * 2}, Load::make(c, {x * 2 + 1}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 1, 10, Store::make(c, {x * 2}, Load::make(c, {x * 2 - 1}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(c, {x * 2}, Load::make(c, {x * 2 + 2}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 1, 10, Store::make(c, {x * 2}, Load::make(c, {x * 2 - 2}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(c, {x * 2}, Load::make(c, {x * 2 + 4}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(c, {x * 6}, Load::make(c, {x * 6 + 5}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(c, {x * 11}, Load::make(c, {x + 1}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(c, {x * 2}, Load::make(c, {x * 3 + 1}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt =
        For::make(x, 0, 10, Store::make(c, {x}, Load::make(c, {x + 5}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt =
        For::make(x, 0, 10, Store::make(c, {x}, Load::make(c, {x + 10}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt =
        Block::make({For::make(
                         x,
                         0,
                         10,
                         Block::make({Store::make(c, {x * 2}, 0, 1),
                                      Store::make(c, {x * 2 + 1}, 1, 1)})),
                     Store::make(b, {0}, Load::make(c, {2}, 1), 1)});
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }
}

void testMemDependencyCheckerLoopDependents() {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        y,
        0,
        i,
        For::make(
            x,
            j,
            k,
            Block::make({Store::make(
                a, {i}, Add::make(Load::make(a, {i}, 1), x), 1)})));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Store::make(b, {x}, Load::make(b, {ExprHandle(9) - x}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt =
        For::make(x, 0, k, Store::make(c, {x / 2}, Load::make(c, {x}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x, 0, k, Store::make(c, {x / 2}, Load::make(c, {x / 2}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        k,
        Store::make(
            c, {Mod::make(x, 2)}, Load::make(c, {Mod::make(x, 2)}, 1), 1));
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }
}

void testMemDependencyCheckerLoopDistinctStrides() {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    mem_dependency::RWDependencyAnalysis analyzer;
    Stmt* stmt = Block::make({
        For::make(
            x,
            0,
            10,
            Store::make(c, {x * 2 + 1}, Load::make(a, {x * 2 + 1}, 1), 1)),
        For::make(
            x, 0, 10, Store::make(c, {x * 2}, Load::make(b, {x * 2}, 1), 1)),
        Store::make(a, {0}, Load::make(c, {3}, 1), 1),
        Store::make(b, {0}, Load::make(c, {3}, 1), 1),
        For::make(x, 0, 10, Store::make(a, {x}, Load::make(c, {x}, 1), 1)),

    });
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }
}

void testMemDependencyGEMM() {
  KernelScope ks;
  int M = 1024;
  int N = 1024;
  int K = 2048;

  Placeholder AP(BufHandle("A", {M, K}, kFloat));
  Placeholder BP(BufHandle("B", {K, N}, kFloat));
  Tensor* CT = Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* m = loops[0];
    For* mo;
    For* mi;
    loop.splitWithMask(m, 4, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* n = loops[2];
    For* no;
    For* ni;
    loop.splitWithMask(n, 16, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* mi = loops[1];
    For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* ni = loops[3];
    For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* mi = loops[2];
    For* k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    loop.cacheAccesses(CT->buf(), "C_regs", loops[2]);
  }

  // loop.prepareForCodegen();
  Stmt* stmt = IRSimplifier::simplify(loop.root_stmt());
  mem_dependency::RWDependencyAnalysis analyzer;
  stmt->accept(&analyzer);

  std::cout << *stmt << "\n";
  for (auto& wi : analyzer.getHistory()) {
    wi->print();
  }
}

} // namespace jit
} // namespace torch
