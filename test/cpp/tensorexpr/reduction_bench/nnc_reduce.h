#pragma once

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

using namespace torch::jit::tensorexpr;

struct NNCContext {
  NNCContext(const std::vector<Tensor*>& output_tensors)
      : program(output_tensors) {}
  LoopNest program;
  std::unique_ptr<CudaCodeGen> codegen;
};

NNCContext get_2dcol_nnc_context(int M, int N) {
  Buffer buf(BufHandle("Input", {M, N}, kFloat));
  Tensor* c = Reduce("sum", {{M, "m"}}, Sum(), buf, {{N, "n"}});
  NNCContext context({c});
  auto loops = NodeFinder<For>::find(context.program.root_stmt());
  context.program.setGPUBlockIndex(loops[0], 0);

  // TODO put transforms here...
  context.program.prepareForCodegen();
  Stmt* root_stmt = IRSimplifier::simplify(context.program.root_stmt());
  std::vector<CodeGen::BufferArg> args = {buf, c};
  context.codegen = std::make_unique<CudaCodeGen>(root_stmt, args);
  return context;
}

NNCContext get_2drow_nnc_context(int M, int N) {
  Buffer buf(BufHandle("Input", {M, N}, kFloat));
  Tensor* c = Reduce(
      "sum",
      {{N, "n"}},
      Sum(),
      [&](const auto& n, const auto& m) { return buf(m, n); },
      {{M, "m"}});
  NNCContext context({c});
  auto loops = NodeFinder<For>::find(context.program.root_stmt());
  context.program.setGPUBlockIndex(loops[0], 0);

  // TODO put transforms here...
  context.program.prepareForCodegen();
  Stmt* root_stmt = IRSimplifier::simplify(context.program.root_stmt());
  std::vector<CodeGen::BufferArg> args = {buf, c};
  context.codegen = std::make_unique<CudaCodeGen>(root_stmt, args);
  return context;
}

void nnc_reduce(NNCContext& context, float* input, float* output) {
  context.codegen->call({input, output});
}

