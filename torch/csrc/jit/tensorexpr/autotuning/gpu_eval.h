#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/arg_library.h>
#include <torch/csrc/jit/tensorexpr/autotuning/autotune_base.h>
#include <torch/csrc/jit/tensorexpr/autotuning/nnc_transforms.h>

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

struct GPUCandidate : public Candidate<LoopNest> {
  using Candidate<LoopNest>::Candidate;
  std::unique_ptr<CudaCodeGen> codegen;

  int nextBlockIdx{0};
  int nextThreadIdx{0};

  GPUCandidate(Program* p, GPUCandidate* pr) : Candidate(p, pr) {
    nextBlockIdx = pr->nextBlockIdx;
    nextThreadIdx = pr->nextThreadIdx;
  }

  GPUCandidate(Program* p, GPUCandidate* pr, Transform* new_transform)
      : Candidate(p, pr, new_transform) {
    nextBlockIdx = pr->nextBlockIdx;
    nextThreadIdx = pr->nextThreadIdx;
  }
};

class GPUPlatform : public Platform<GPUCandidate> {
 public:
  ArgumentLibrary argLibrary_;
  LoopNest* program_;
  GPUCandidate original;
  std::vector<SizedBuffer> deviceBuffers_;

  GPUPlatform(
      TuningStats& stats,
      LoopNest* program,
      const std::vector<CodeGen::BufferArg>& args)
      : Platform(stats, program, args),
        argLibrary_(program, args),
        program_(program),
        original(program) {
    // Set up device buffers for inputs/outputs.
    for (auto& a : argLibrary_.args()) {
      int* bdev = nullptr;
      size_t size = argLibrary_.getArgBytes(a.var());
      cudaMalloc(&bdev, size);
      deviceBuffers_.emplace_back((void*)bdev, size);
    }
    // Set the reference output values based on an untransformed program.
    for (unsigned int i = 0; i < argLibrary_.numArgsets(); ++i) {
      auto args = argLibrary_.getCallArgs(i);
      runCandidates({&original});
      if (original.times_run == 0) {
        std::cout << "bad input program!\n";
      }
      argLibrary_.setReferenceArgs(i, std::move(args));
    }
  }

  void bindInitialAxes() {
    try {
      auto initialLoops = NodeFinder<For>::find(original.program->root_stmt());
      // TODO check this binding works, if we break the original canidate we're
      // stuck.

      if (initialLoops.size() > 1) {
        original.program->setGPUBlockIndex(initialLoops[1], 0);
        original.schedule.append(new transforms::BindBlockIdx(
            initialLoops[1]->var()->name_hint(), original.nextBlockIdx++));
      }

      if (initialLoops.size() > 0) {
        original.program->setGPUThreadIndex(initialLoops[0], 0);
        original.schedule.append(new transforms::BindThreadIdx(
            initialLoops[0]->var()->name_hint(), original.nextThreadIdx++));
      }
    } catch (std::exception& e) {
      std::cout << "expection thrown while attempting binding of original axes"
                << e.what() << "\n";
    }
  }

  GPUCandidate* getOriginalCandidate() {
    return &original;
  }

  void runCandidates(std::vector<GPUCandidate*> cohort) {
    std::cout << "runCandidates: " << cohort.size() << " to run\n";
    // TODO: could spin these out to different threads
    for (auto* c : cohort) {
      GPUCandidate* candidate = dynamic_cast<GPUCandidate*>(c);

      // another clone because we want the candidate to stay
      // un-preparedForCodegen.
      LoopNest temp(*candidate->program);
      temp.prepareForCodegen();

      try {
        if (!candidate->codegen) {
          Stmt* s = IRSimplifier::simplify(temp.root_stmt());
          auto start = host_timestamp();
          candidate->codegen =
              std::make_unique<CudaCodeGen>(s, argLibrary_.args());
          stats_.codegen_time += TO_MS(host_timestamp() - start);
        }
      } catch (std::exception& e) {
        std::cout << "failed codegen: " << e.what() << "\n";
        std::cout << *temp.root_stmt();
        return;
      }

      // TODO magic number
      for (int i = 0; i < 3; ++i) {
        std::vector<CodeGen::CallArg> hostArgs =
            argLibrary_.getCallArgs(candidate->times_run);
        std::vector<CodeGen::CallArg> deviceArgs;

        for (int i = 0; i < hostArgs.size(); ++i) {
          void* host = hostArgs[i].data();
          void* device = deviceBuffers_[i].ptr;
          deviceArgs.push_back(CodeGen::CallArg(device));
          cudaMemcpy(
              device, host, deviceBuffers_[i].len, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        cudaEvent_t start, stop;

        stats_.total_runs++;
        try {
          cudaEventCreate(&start);
          cudaEventCreate(&stop);

          cudaEventRecord(start);
          candidate->codegen->call(deviceArgs);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
        } catch (std::exception& e) {
          return;
        }
        candidate->times_run++;

        int idx = 0;
        for (auto& a : argLibrary_.args()) {
          if (argLibrary_.isOutput(a.var())) {
            void* host = hostArgs[idx].data();
            void* device = deviceBuffers_[idx].ptr;
            cudaMemcpy(
                host, device, deviceBuffers_[idx].len, cudaMemcpyDeviceToHost);
          }
          idx++;
        }
        cudaDeviceSynchronize();
        candidate->valid =
            argLibrary_.checkOutputs(candidate->times_run - 1, hostArgs);
        if (!candidate->valid) {
          break;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        size_t dur = milliseconds * 1000;

        size_t total_us =
            (candidate->times_run * candidate->resolved_cost) + dur;
        stats_.running_time += std::chrono::milliseconds((int)milliseconds);
        candidate->resolved_cost = total_us / candidate->times_run;
      }
    }
  }
};

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch
