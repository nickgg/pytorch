#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class BoundsInference : public IRVisitor {
 public:
  void visit(const FunctionCall* v) override;
  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const ReduceOp* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;

  BoundsInfo accesses() const {
    return accesses_;
  }

 private:
  BoundsInfo accesses_;
};

void BoundsInference::visit(const Load* v) {
  accesses_[v->buf()].push_back({kLoad, v->indices(), v->indices()});
}

void BoundsInference::visit(const FunctionCall* v) {
  accesses_[v->tensor()->buf()].push_back({kLoad, v->params(), v->params()});
}

void BoundsInference::visit(const Store* v) {
  accesses_[v->buf()].push_back({kStore, v->indices(), v->indices()});
  IRVisitor::visit(v);
}

void BoundsInference::visit(const ReduceOp* v) {
  accesses_[v->accumulator()].push_back(
      {kLoad, v->output_args(), v->output_args()});
  IRVisitor::visit(v);
}

void BoundsInference::visit(const For* v) {
  v->body()->accept(this);
  for (auto& pair : accesses_) {
    for (TensorAccessBoundsInfo& access : pair.second) {
      for (size_t j = 0; j < access.start.size(); j++) {
        // TODO: This function assumes that all indices grow monotonically and
        // thus for the loop:
        //   for i in A..B:
        //     buf[i] = i
        // the range for i is [A, B). It should be generalized to correctly
        // handle all cases.
        const Expr* old_start = access.start[j];
        const Expr* old_stop = access.stop[j];
        const Expr* new_start = Substitute(old_start, {{v->var(), v->start()}});
        const Expr* new_stop = Substitute(
            old_stop, {{v->var(), new Sub(v->stop(), new IntImm(1))}});

        access.start[j] = IRSimplifier::simplify(new_start);
        access.stop[j] = IRSimplifier::simplify(new_stop);
      }
    }
  }
}

void BoundsInference::visit(const Block* v) {
  BoundsInfo res;
  for (auto s : *v) {
    s->accept(this);
    for (auto& pair : accesses_) {
      res[pair.first].insert(
          res[pair.first].end(), pair.second.begin(), pair.second.end());
    }
  }
  accesses_ = res;
}

void printBoundsInfo(const BoundsInfo& v) {
  std::cerr << "Access vector {\n";
  for (auto& pair : v) {
    std::cerr << *pair.first << " in [";
    bool first = true;
    for (const auto& b : pair.second) {
      if (!first) {
        std::cerr << ", ";
      }
      std::cerr << ((b.kind == kLoad) ? "LOAD" : "STORE") << "(";
      int i = 0;
      if (b.start.empty()) {
        std::cerr << "0";
      }
      for (const auto& s : b.start) {
        if (i != 0) {
          std::cerr << ", ";
        }
        std::cerr << *s;
        i++;
      }
      std::cerr << "; ";
      i = 0;
      if (b.stop.empty()) {
        std::cerr << "0";
      }
      for (const auto& s : b.stop) {
        if (i != 0) {
          std::cerr << ", ";
        }
        std::cerr << *s;
        i++;
      }
      std::cerr << ")";
      first = false;
    }
    std::cerr << "]\n";
  }
  std::cerr << "}\n";
}

/*
 * Go through the given BoundsInfo vector and merge entries corresponding to
 * the same buf. E.g. given
 *    [{a, kLoad, 0, 100}, {b, kStore, 0, 100}, {a, kLoad, 10, 110}]
 * produce:
 *    [{a, kLoad, 0, 110}, {b, kStore, 0, 100}]
 */
BoundsInfo mergeTensorAccesses(const BoundsInfo& unmerged) {
  BoundsInfo res;
  // For each buf in the BoundsInfo:
  for (auto& pair : unmerged) {
    const std::vector<TensorAccessBoundsInfo>& new_vec = pair.second;
    std::vector<TensorAccessBoundsInfo>& existing_vec = res[pair.first];

    // For each bound pair in the unmerged set:
    for (const auto& new_bound : new_vec) {
      bool found = false;
      // For each already merged bound pair:
      for (auto& existing_bound : existing_vec) {
        // Only merge the same kind of access.
        if (existing_bound.kind != new_bound.kind) {
          continue;
        }

        // Sanity check the buf indices have the same dimensionality.
        TORCH_INTERNAL_ASSERT(new_bound.start.size() == new_bound.stop.size());
        TORCH_INTERNAL_ASSERT(
            existing_bound.start.size() == existing_bound.stop.size());
        TORCH_INTERNAL_ASSERT(
            new_bound.start.size() == existing_bound.start.size());

        std::vector<const Expr*> start;
        std::vector<const Expr*> stop;
        bool fail = false;
        // For each dimension:
        for (size_t i = 0; i < new_bound.start.size(); ++i) {
          // The range of the new bound must overlap the existing bound.
          // TODO(nickg): we allow all dimensions to partially overlap,
          // which will overstate the bounds.
          // auto pair = mem_dependency::rangeOverlap(
          //     new_bound.start[i],
          //     new_bound.stop[i],
          //     existing_bound.start[i],
          //     existing_bound.stop[i],
          //     true);
          // if (pair.first == nullptr) {
          //   fail = true;
          //   break;
          // }
          // start.push_back(pair.first);
          // stop.push_back(pair.second);
        }
        if (fail) {
          continue;
        }
        found = true;
        // Update the existing bound.
        existing_bound.start = start;
        existing_bound.stop = stop;
      }
      if (!found) {
        existing_vec.push_back(new_bound);
      }
    }
  }

  return res;
}

BoundsInfo inferBounds(Stmt* s) {
  BoundsInference ac;
  s->accept(&ac);
  return mergeTensorAccesses(ac.accesses());
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
