#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/autotune_base.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

template <typename CandidateType>
class ResultLibrary {
 public:
  ResultLibrary(TuningStats& stats) : stats_(stats) {}

  bool insertCandidate(CandidateType* candidate) {
    if (!candidate->valid) {
      return false;
    }
    if (!candidatesByHash_
             .emplace(
                 simplifier_.hash(candidate->program->root_stmt()), candidate)
             .second) {
      return false;
    }
    resolvedCandidates_.push_back(candidate);
    sortResolved();
    return true;
  }

  size_t insertBatch(std::vector<CandidateType*>& candidates) {
    size_t inserted = 0;
    for (auto* c : candidates) {
      if (!c->valid) {
        continue;
      }
      if (c->times_run > 0 &&
          candidatesByHash_
              .emplace(simplifier_.hash(c->program->root_stmt()), c)
              .second) {
        resolvedCandidates_.push_back(c);
        inserted++;
      }
    }
    if (inserted > 0) {
      sortResolved();
    }
    return inserted;
  }

  void filterNewCandidates(std::vector<CandidateType*>& candidates) {
    candidates.erase(candidates.remove_if(
        candidates.begin(), candidates.end(), [&](CandidateType* c) {
          return candidatesByHash_.count(
                     simplifier_.hash(c->program->root_stmt())) != 0;
        }));
  }

  std::vector<CandidateType*> getBestResolvedCandidates(ssize_t num) {
    std::vector<CandidateType*> res;
    res.reserve(num);
    for (auto* c : resolvedCandidates_) {
      if (num-- <= 0) {
        break;
      }
      res.push_back(c);
    }

    return res;
  }

  std::deque<CandidateType*>& getAllCandidates() {
    return resolvedCandidates_;
  }

  SimplifierHashType hash(const Expr* e) {
    return simplifier_.hash(e);
  }

  SimplifierHashType hash(Stmt* s) {
    return simplifier_.hash(s);
  }

 protected:
  std::deque<CandidateType*> resolvedCandidates_;
  std::unordered_map<SimplifierHashType, CandidateType*> candidatesByHash_;
  IRSimplifier simplifier_;
  TuningStats& stats_;

  void sortResolved() {
    auto start = host_timestamp();
    std::sort(
        resolvedCandidates_.begin(),
        resolvedCandidates_.end(),
        [](const CandidateType* a, const CandidateType* b) -> bool {
          if (a->resolved_cost == b->resolved_cost) {
            if (a->times_run == b->times_run) {
              return a > b;
            }
            return a->times_run > b->times_run;
          }
          return a->resolved_cost < b->resolved_cost;
        });
    stats_.sorting_time += TO_MS(host_timestamp() - start);
  }
};

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch

