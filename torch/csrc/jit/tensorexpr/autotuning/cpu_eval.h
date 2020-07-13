#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/arg_library.h>
#include <torch/csrc/jit/tensorexpr/autotuning/autotune_base.h>

#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

struct CPUCandidate : public Candidate<LoopNest> {
  using Candidate<LoopNest>::Candidate;
  std::unique_ptr<LLVMCodeGen> codegen;
};

class CPUPlatform : public Platform<CPUCandidate> {
 public:
  ArgumentLibrary argLibrary_;
  LoopNest* program_;
  CPUCandidate original;

  CPUPlatform(
      TuningStats& stats,
      LoopNest* program,
      const std::vector<CodeGen::BufferArg>& args)
      : Platform(stats, program, args),
        argLibrary_(program, args),
        program_(program),
        original(program) {
    // Set the reference output values based on an untransformed program.
    for (unsigned int i = 0; i < argLibrary_.numArgsets(); ++i) {
      auto args = argLibrary_.getCallArgs(i);
      runCandidates({&original});
      argLibrary_.setReferenceArgs(i, std::move(args));
    }
  }

  CPUCandidate* getOriginalCandidate() {
    return &original;
  }

  void runCandidates(std::vector<CPUCandidate*> cohort) {
    std::cout << "runCandidates: " << cohort.size() << " to run\n";
    // TODO: could spin these out to different threads
    for (auto* c : cohort) {
      CPUCandidate* candidate = dynamic_cast<CPUCandidate*>(c);

      // another clone because we want the candidate to stay
      // un-preparedForCodegen.
      LoopNest temp(*candidate->program);
      temp.prepareForCodegen();

      if (!candidate->codegen) {
        Stmt* s = IRSimplifier::simplify(temp.root_stmt());
        auto start = host_timestamp();
        candidate->codegen =
            std::make_unique<LLVMCodeGen>(s, argLibrary_.args());
        stats_.codegen_time += TO_MS(host_timestamp() - start);
      }

      stats_.total_runs++;
      candidate->times_run++;
      std::vector<CodeGen::CallArg> runArgs =
          argLibrary_.getCallArgs(candidate->times_run);

      auto start = host_timestamp();
      candidate->codegen->call(runArgs);
      auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
          host_timestamp() - start);

      size_t total_us =
          (candidate->times_run * candidate->resolved_cost) + dur.count();
      stats_.running_time +=
          std::chrono::duration_cast<std::chrono::milliseconds>(dur);
      candidate->resolved_cost = total_us / candidate->times_run;
    }
  }
};

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch
