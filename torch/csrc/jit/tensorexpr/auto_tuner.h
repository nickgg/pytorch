#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <chrono>
#include <random>

namespace torch {
namespace jit {
namespace tensorexpr {

namespace tuning {

struct Transform {
  virtual void log() const {
    std::cout << "unimplemented\n";
  };

  virtual std::string name() const {
    return "null";
  }

  virtual Transform* clone() const = 0;
};

struct Vectorize : public Transform {
  std::string loopVar;
  Vectorize(std::string l) : loopVar(l) {}

  void log() const override {
    std::cout << "Vectorize(" << loopVar << ")";
  }

  std::string name() const override {
    return "Vectorize";
  }

  Transform* clone() const override {
    return new Vectorize(loopVar);
  }
};

struct SplitWithMask : public Transform {
  std::string loopVar;
  int factor;
  SplitWithMask(std::string l, int f) : loopVar(l), factor(f) {}

  void log() const override {
    std::cout << "SplitWithMask(" << loopVar << ", " << factor << ")";
  }

  std::string name() const override {
    return "SplitWithMask";
  }

  Transform* clone() const override {
    return new SplitWithMask(loopVar, factor);
  }
};

struct SplitWithTail : public Transform {
  std::string loopVar;
  int factor;
  SplitWithTail(std::string l, int f) : loopVar(l), factor(f) {}

  void log() const override {
    std::cout << "SplitWithTail(" << loopVar << ", " << factor << ")";
  }

  std::string name() const override {
    return "SplitWithTail";
  }

  Transform* clone() const override {
    return new SplitWithTail(loopVar, factor);
  }
};

struct ReorderAxis : public Transform {
  std::string loop1;
  std::string loop2;
  ReorderAxis(std::string l1, std::string l2) : loop1(l1), loop2(l2) {}

  void log() const override {
    std::cout << "ReorderAxis(" << loop1 << ", " << loop2 << ")";
  }

  std::string name() const override {
    return "ReorderAxis";
  }

  Transform* clone() const override {
    return new ReorderAxis(loop1, loop2);
  }
};

struct Inline : public Transform {
  std::string buf;
  Inline(std::string b) : buf(b) {}

  void log() const override {
    std::cout << "Inline(" << buf << ")";
  }

  std::string name() const override {
    return "Inline";
  }

  Transform* clone() const override {
    return new Inline(buf);
  }
};

struct Rfactor : public Transform {
  std::string buf;
  std::string var;
  Rfactor(std::string b, std::string v) : buf(b), var(v) {}

  void log() const override {
    std::cout << "Rfactor(" << buf << ", " << var << ")";
  }

  std::string name() const override {
    return "Rfactor";
  }

  Transform* clone() const override {
    return new Rfactor(buf, var);
  }
};

struct ComputeAt : public Transform {
  int stmt_idx;
  std::string loopVar;
  ComputeAt(int s, std::string l) : stmt_idx(s), loopVar(l) {}

  void log() const override {
    std::cout << "ComputeAt(" << stmt_idx << ", " << loopVar << ")";
  }

  std::string name() const override {
    return "ComputeAt";
  }

  Transform* clone() const override {
    return new ComputeAt(stmt_idx, loopVar);
  }
};

} // namespace tuning

class TORCH_API AutoTuner {
  class Candidate {
   public:
    Candidate(const LoopNest& n) : loopnest(n) {}
    Candidate(
        const LoopNest& n,
        std::vector<tuning::Transform*>& other_schedule)
        : loopnest(n) {
      for (auto* t : other_schedule) {
        schedule.push_back(t->clone());
      }
    }

    void logSchedule() {
      std::cout << "[";
      bool first = true;
      for (auto* t : schedule) {
        if (!first) {
          std::cout << ", ";
        }
        t->log();
        first = false;
      }
      std::cout << "]";
    }

    int depth() {
      return schedule.size();
    }

    LoopNest loopnest;
    std::vector<tuning::Transform*> schedule;
    int runs{0};
    std::chrono::microseconds time{std::chrono::microseconds::max()};
    bool children_generated{false};
    SimplifierHashType hash;

    std::unique_ptr<CodeGen> codegen;
  };

 public:
  AutoTuner(
      LoopNest& loopnest,
      const std::vector<CodeGen::BufferArg>& args,
      at::Device device = at::kCPU)
      : rootNest_(loopnest), args_(args), expander(&simplifier) {
    GenerateCallArgs();
  }

  ~AutoTuner();

  enum RunResult {
    SUCCESS = 0,
    CODEGEN_FAIL = 1,
    RUN_FAIL = 2,
    BAD_OUTPUTS = 3
  };
  bool first_{true};

  void run(int iterations = 0);
  LoopNest getBestCandidate();

  // private:
  LoopNest& rootNest_;
  std::unordered_map<const Var*, std::vector<CodeGen::CallArg>> argData_;
  const std::vector<CodeGen::BufferArg> args_;

  struct SizedBuffer {
    void* ptr{nullptr};
    size_t len{0};
  };

  std::unordered_map<const Var*, SizedBuffer> outputs_;
  std::unordered_map<const Var*, SizedBuffer> referenceOutputs_;
  bool referenceReady_{false};

  // Generated schedules by schedule depth
  std::vector<std::deque<Candidate*>> potential_candidates_;
  std::deque<Candidate*> resolved_candidates_;
  std::unordered_map<SimplifierHashType, Candidate*> candidatesByHash_;

  std::string codegenName_{"llvm_codegen"};

  void GenerateCallArgs();

  void generateSplitWithTail(Candidate* c, int factor);
  void generateSplitWithMask(Candidate* c, int factor);
  void generateVectorize(Candidate* c);
  void generateReorder(Candidate* c);
  void generateInlining(Candidate* c);
  void generateRfactor(Candidate* c);
  // void generateComputeAt(Candidate* c);

  void generateChildren(Candidate* c, bool first);
  void generateNextCandidates();
  std::deque<Candidate*> pickCandidates(size_t num);
  bool addPotentialCandidate(Candidate* c);
  SimplifierHashType hashCandidateStmt(Stmt* s);
  void sortCandidates();
  bool checkOutputs();
  RunResult runCandidate(Candidate* c);

  PolynomialTransformer simplifier;
  TermExpander expander;

  int generated_{0};
  int tested_{0};

  std::chrono::milliseconds running_time{0};
  std::chrono::milliseconds codegen_time{0};
  std::chrono::milliseconds hashing_time{0};
  std::chrono::milliseconds simplify_time{0};
  std::chrono::milliseconds generation_time{0};
  std::chrono::milliseconds sorting_time{0};
  std::chrono::milliseconds checking_time{0};

  struct RunStats {
    int success{0};
    int codegen_fail{0};
    int run_fail{0};
    int bad_output{0};
  };

  RunStats runStats_;
  std::map<std::string, RunStats> perOpStats_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
