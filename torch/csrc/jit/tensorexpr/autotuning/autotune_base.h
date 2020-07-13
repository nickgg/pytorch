#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/tuner_stats.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>

#include <iostream>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

struct Transform {
  virtual void log() const {};

  virtual std::string name() const {
    return "null";
  }

  virtual Transform* clone() const = 0;
};

class Schedule {
 public:
  void log() {
    std::cout << "[";
    bool first = true;
    for (auto* t : transforms) {
      if (!first) {
        std::cout << ", ";
      }
      t->log();
      first = false;
    }
    std::cout << "]";
  }

  Schedule clone() {
    Schedule next;
    for (auto* t : transforms) {
      next.transforms.push_back(t->clone());
    }
    return next;
  }

  void append(Transform* t) {
    transforms.push_back(t);
  }

  size_t depth() {
    return transforms.size();
  }

  std::vector<Transform*> transforms;
};

template <class Program_>
class Candidate {
 public:
  using Program = Program_;
  Candidate(Program* p) : program(p) {}
  Candidate(Program* p, Candidate<Program>* pr)
      : program(p), schedule(pr->schedule.clone()), parent(pr) {
    pr->children.push_back(this);
  }

  Candidate(Program* p, Candidate<Program>* pr, Transform* new_transform)
      : Candidate(p, pr) {
    // This calls the 2 arg ctor, which will add this to the children of pr.
    schedule.append(new_transform);
  }
  virtual ~Candidate() {}

  Program* program;
  Schedule schedule;
  bool valid{true};

  size_t cost_estimate{0};

  size_t resolved_cost{0};
  size_t times_run{0};

  Candidate<Program>* parent{nullptr};
  std::vector<Candidate<Program>*> children;
};

template <class CandidateType>
class Platform {
 protected:
  TuningStats& stats_;

 public:
  Platform(
      TuningStats& stats,
      typename CandidateType::Program* p,
      const std::vector<CodeGen::BufferArg>& args)
      : stats_(stats) {}
  virtual void runCandidates(std::vector<CandidateType*> cohort) = 0;

  const TuningStats& getStats() {
    return stats_;
  }
};

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch
