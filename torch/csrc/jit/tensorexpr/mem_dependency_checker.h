#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <deque>
#include <vector>

#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace mem_dependency {

enum class Access {
  Load,
  VarRef,
  Store,
  Let,
  For,
  Allocate,
  AtomicAdd,
  ReduceOp
};
const char* AccessToString(Access a) {
  switch (a) {
    case Access::Load:
      return "Load";
    case Access::VarRef:
      return "VarRef";
    case Access::Store:
      return "Store";
    case Access::Let:
      return "Let";
    case Access::For:
      return "For";
    case Access::Allocate:
      return "Allocate";
    case Access::AtomicAdd:
      return "AtomicAdd";
    case Access::ReduceOp:
      return "ReduceOp";
    default:
      break;
  }
  return "Unknown";
}

struct TORCH_API Bound {
  const Expr* start{nullptr};
  const Expr* end{nullptr};
  const Expr* stride{nullptr};

  Bound() = default;
  Bound(const Expr* s, const Expr* e, const Expr* st)
      : start(s), end(e), stride(st) {}

  void print() {
    std::cout << "(" << *start << ", " << *end << ", " << *stride << ")";
  }
};
enum OverlapKind {
  NoOverlap = 0,
  PartialOverlap,
  TotalOverlapA,
  TotalOverlapB
};
OverlapKind TORCH_API boundOverlap(Bound a, Bound b);

using IndexBounds = std::vector<Bound>;
using VarBoundMap = std::unordered_map<const Var*, Bound>;

struct AccessInfo {
  AccessInfo(size_t i, Access t, const Stmt* s, const Var* v, IndexBounds b)
      : id(i), type(t), stmt(s), expr(nullptr), var(v), bounds(b) {}

  AccessInfo(
      size_t i,
      Access t,
      const Expr* e,
      const Stmt* s,
      const Var* v,
      IndexBounds b)
      : id(i), type(t), stmt(s), expr(e), var(v), bounds(b) {}

  size_t id;
  Access type;
  const Stmt* stmt;

  // TODO exprs are not unique and may be owned by multiple stmts, what do?
  const Expr* expr;
  const Var* var;
  IndexBounds bounds;

  // Yes these should be sorted.
  std::map<size_t, std::shared_ptr<AccessInfo>> dependencies;
  std::map<size_t, std::shared_ptr<AccessInfo>> dependents;

  std::vector<const Expr*> getIndices() {
    std::vector<const Expr*> indices;

    if (expr) {
      if (const Load* load = dynamic_cast<const Load*>(expr)) {
        indices = load->indices();
      } else if (const ReduceOp* reduce = dynamic_cast<const ReduceOp*>(expr)) {
        indices = reduce->output_args();
      }
    } else {
      if (const Store* store = dynamic_cast<const Store*>(stmt)) {
        indices = store->indices();
      }
    }
    return indices;
  }

  virtual void print() {
    std::cout << id << ". " << AccessToString(type) << ": " << *var << "[";
    if (bounds.size() > 0) {
      for (int i = 0; i < bounds.size() - 1; ++i) {
        bounds[i].print();
        std::cout << ", ";
      }

      int i = bounds.size() - 1;
      bounds[i].print();
    }
    std::cout << "]";

    if (!dependencies.empty()) {
      std::cout << " - depends on: ";
      for (auto it = dependencies.begin(); it != dependencies.end(); ++it) {
        std::cout << it->second->id << " ";
      }
    }

    if (!dependents.empty()) {
      std::cout << " - dependents: ";
      for (auto it = dependents.begin(); it != dependents.end(); ++it) {
        std::cout << it->second->id << " ";
      }
    }

    std::cout << "\n";
  }

  void addDependency(std::shared_ptr<AccessInfo> write) {
    dependencies[write->id] = write;
  }

  void addDependent(std::shared_ptr<AccessInfo> read) {
    dependents[read->id] = read;
  }

  bool hasDependency(const std::shared_ptr<AccessInfo>& info) {
    return dependencies.count(info->id) != 0;
  }

  bool isRead() const {
    switch (type) {
      case Access::Load:
        return true;
      case Access::VarRef:
        return true;
      case Access::For:
        return true;
      case Access::ReduceOp:
        return true;
      default:
        break;
    }
    return false;
  }

  bool isWrite() const {
    switch (type) {
      case Access::Store:
        return true;
      case Access::Let:
        return true;
      case Access::For:
        return true;
      case Access::Allocate:
        return true;
      case Access::AtomicAdd:
        return true;
      default:
        break;
    }
    return false;
  }
};

using BoundRelationship = std::pair<IndexBounds, std::shared_ptr<AccessInfo>>;

struct Scope {
  Scope(Block* b, std::shared_ptr<Scope> p) : block(b), parent(p) {}

  Block* block;
  std::shared_ptr<Scope> parent;

  std::unordered_map<const Var*, Bound> shadowedVarBounds;
  std::unordered_set<const Var*> localVars;

  std::deque<std::shared_ptr<AccessInfo>> accesses_;

  std::unordered_map<const Var*, std::deque<BoundRelationship>> openWrites_;
};

Bound normalizeBound(Bound a);
std::vector<Bound> subtractBound(Bound a, Bound b);

OverlapKind overlaps(const IndexBounds& a, const IndexBounds& b) {
  // All accesses to a buf must have the same dimensionality.
  TORCH_INTERNAL_ASSERT(a.size() == b.size());

  OverlapKind overlap = NoOverlap;
  for (size_t i = 0; i < a.size(); ++i) {
    OverlapKind bOverlap = boundOverlap(a[i], b[i]);
    if (bOverlap == NoOverlap) {
      return NoOverlap;
    }

    if (bOverlap == PartialOverlap) {
      overlap = PartialOverlap;
      break;
    }

    if (bOverlap == NoOverlap) {
      overlap = bOverlap;
    } else if (bOverlap != overlap) {
      overlap = PartialOverlap;
      break;
    }
  }

  return overlap;
}

// recursively takes dimensional slices from slices and adds them to each
// segment in segments.
//
// E.g. [[(0, 10, 1), (15, 20, 1)],
//       [(0, 10, 1)],
//       [(1, 2, 1), (5, 6, 1)]]
//  =>  [[(0, 10, 1), (0, 10, 1), (1, 2, 1)],
//       [(0, 10, 1), (0, 10, 1), (5, 6, 1)],
//       [(15, 20, 1), (0, 10, 1), (1, 2, 1)],
//       [(15, 20, 1), (0, 10, 1), (5, 6, 1)]]
//
void boundPermutation(
    std::deque<IndexBounds>& slices,
    std::vector<IndexBounds>& segments) {
  if (slices.empty()) {
    return;
  }

  if (segments.empty()) {
    for (auto& bound : slices.front()) {
      segments.push_back({bound});
    }
  } else {
    std::vector<IndexBounds> seg_copy;
    segments.swap(seg_copy);

    for (auto& seg : seg_copy) {
      for (auto& bound : slices.front()) {
        auto newSeg = seg;
        newSeg.push_back(bound);
        segments.push_back(newSeg);
      }
    }
  }

  slices.pop_front();
  boundPermutation(slices, segments);
}

// Returns the bound slices created by subtracting the IndexBounds B from A.
std::vector<IndexBounds> subtractIndicesBounds(
    IndexBounds& A,
    IndexBounds& B,
    OverlapKind overlap) {
  if (overlap == NoOverlap) {
    return {A};
  }

  if (overlap == TotalOverlapA) {
    return {};
  }
  // All accesses to a buf must have the same dimensionality.
  TORCH_INTERNAL_ASSERT(A.size() == B.size());

  // Each dimension can be sliced into multiple bound segments.
  std::deque<IndexBounds> boundSlices;

  for (size_t i = 0; i < A.size(); ++i) {
    // std::cout << "SIB " << overlap << " ";
    // A[i].print();
    // std::cout << " - ";
    // B[i].print();
    // std::cout << " = ";
    auto slice = subtractBound(A[i], B[i]);
    // for (auto& s : slice) {
    //   s.print();
    // }
    // std::cout << "\n";
    boundSlices.push_back(slice);
  }

  for (auto& v : boundSlices) {
    // std::cout << "Slice: ";
    // for (auto& b : v) {
    //   b.print();
    // }
    // std::cout << "\n";
    // Actually total overlap... should have caught this?
    if (v.empty()) {
      return {};
    }
  }

  std::vector<IndexBounds> segments;
  boundPermutation(boundSlices, segments);
  return segments;
}

class TORCH_API RWDependencyAnalysis : public IRVisitor {
 public:
  RWDependencyAnalysis() {
    currentScope_ = std::make_shared<Scope>(nullptr, nullptr);
    std::cout << "\n\n";
  }
  virtual ~RWDependencyAnalysis() {}

  // void visit(const For* v) override;

  // void visit(const Block* v) override;

  // void visit(const Load* v) override;

  // void visit(const IfThenElse* v) override;

  // void visit(const Cond* v) override;
  const std::deque<std::shared_ptr<AccessInfo>>& getHistory() {
    return currentScope_->accesses_;
  }

 private:
  void visit(const Store* v) override {
    const Stmt* last = lastStmt_;
    lastStmt_ = v;
    v->value()->accept(this);
    lastStmt_ = last;

    const Var* var = v->buf()->base_handle();

    auto info = std::make_shared<AccessInfo>(
        nextAccess_++, Access::Store, v, var, getIndicesBounds(v->indices()));

    // This write is open, and will close any open writes that it totally
    // overlaps.

    auto& history = currentScope_->openWrites_[var];
    updateWriteHistory(history, info);
    currentScope_->accesses_.push_back(info);
  }

  void visit(const Load* v) override {
    const Var* var = v->buf()->base_handle();
    auto load = std::make_shared<AccessInfo>(
        nextAccess_++,
        Access::Load,
        v,
        lastStmt_,
        var,
        getIndicesBounds(v->indices()));

    auto& writeHistory = currentScope_->openWrites_[var];
    updateWriteHistory(writeHistory, load);
    currentScope_->accesses_.push_back(load);
  }

  void visit(const ReduceOp* v) override {
    const Var* var = v->accumulator()->base_handle();
    auto load = std::make_shared<AccessInfo>(
        nextAccess_++,
        Access::ReduceOp,
        v,
        lastStmt_,
        var,
        getIndicesBounds(v->output_args()));

    auto& writeHistory = currentScope_->openWrites_[var];
    updateWriteHistory(writeHistory, load);
    currentScope_->accesses_.push_back(load);

    v->body().node()->accept(this);
  }

  void visit(const Var* v) override {}

  void visit(const For* v) override {
    auto newScope = std::make_shared<Scope>(v->body(), currentScope_);

    const Var* var = v->var();

    currentScope_ = newScope;

    const Stmt* last = lastStmt_;
    lastStmt_ = v;

    v->var()->accept(this);
    v->start()->accept(this);
    v->stop()->accept(this);
    v->body()->accept(this);

    lastStmt_ = last;

    // Ok now we need to determine whether accesses in the loop depend on
    // other loop iterations.
    //
    // This is the real challenge here, it depends on both the fully expanded
    // bounds and the symbolic bounds.

    // The indices must change monotonically to avoid intersection. This is
    // hard to determine, so here's our heuristic I hope it's conservative
    // enough.

    // the size of at least one dependent index must be >= the size of the
    // loop.

    std::vector<std::vector<const Expr*>> loopStrides;
    loopStrides.resize(currentScope_->accesses_.size());

    for (size_t a = 0; a < currentScope_->accesses_.size(); ++a) {
      auto& info = currentScope_->accesses_[a];

      std::vector<const Expr*> indices = info->getIndices();

      std::vector<const Expr*>& loopIndicesStride = loopStrides[a];
      loopIndicesStride.resize(indices.size());
      // store must depend on the loop var in some way.
      for (size_t i = 0; i < indices.size(); i++) {
        VarFinder vf;
        if (vf.find(indices[i]).count(var) == 0) {
          loopIndicesStride[i] = new IntImm(0);
        } else {
          std::cout << " Subbing loopVar " << *var << "\n";
          std::cout << "PRE ";
          for (auto& b : info->bounds) {
            b.print();
          }
          std::cout << "\n";
          info->bounds[i].start = IRSimplifier::simplify(
              Substitute(info->bounds[i].start, {{var, v->start()}}));
          info->bounds[i].end = IRSimplifier::simplify(Substitute(
              info->bounds[i].end, {{var, new Sub(v->stop(), new IntImm(1))}}));

          const Expr* zeroStep = indices[i];
          const Expr* oneStep =
              Substitute(indices[i], {{var, new Add(var, new IntImm(1))}});
          loopIndicesStride[i] =
              IRSimplifier::simplify(new Sub(oneStep, zeroStep));

          const Expr* loopExtent =
              IRSimplifier::simplify(new Sub(v->stop(), v->start()));

          const Expr** existingStride = &info->bounds[i].stride;
          std::cout << "existingStride " << **existingStride
                    << " higherStride: " << *loopIndicesStride[i] << "\n";
          std::cout << "loopExtent: " << *loopExtent << " -> size:  "
                    << *IRSimplifier::simplify(
                           new Mul(loopExtent, loopIndicesStride[i]))
                    << "\n";

          if ((*existingStride)->isConstant() &&
              immediateEquals(*existingStride, 0)) {
            *existingStride = loopIndicesStride[i];
          } else {
            // *existingStride = IRSimplifier::simplify(
            //     new Mul(*existingStride, loopIndicesStride[i]));
          }

          info->bounds[i] = normalizeBound(info->bounds[i]);
          std::cout << "POST ";
          for (auto& b : info->bounds) {
            b.print();
          }
          std::cout << "\n";
        }
      }
    }

    auto equalStrides = [&](const std::vector<const Expr*>& A,
                            const std::vector<const Expr*>& B) {
      if (A.size() != B.size()) {
        return false;
      }
      for (auto i = 0; i < A.size(); ++i) {
        if (!exprEquals(A[i], B[i])) {
          return false;
        }
      }
      return true;
    };

    for (size_t a = 0; a < currentScope_->accesses_.size(); ++a) {
      auto& info = currentScope_->accesses_[a];
      if (!info->isRead()) {
        continue;
      }

      if (!info->dependencies.empty()) {
        // already dependent on a mutation in the loop.
        continue;
      }

      // Scan from the bottom of the loop.
      for (size_t j = currentScope_->accesses_.size() - 1; j > a; --j) {
        std::shared_ptr<AccessInfo> other = currentScope_->accesses_[j];
        if (info->var != other->var) {
          continue;
        }

        bool safe = false;
        for (int b = 0; b < info->bounds.size(); ++b) {
          if (!exprEquals(loopStrides[a][b], loopStrides[j][b])) {
            continue;
          }

          if (loopStrides[a][b]->isConstant() &&
              immediateEquals(loopStrides[a][b], 0)) {
            continue;
          }

          // TODO describe safety logic
          // ((starDiff < 0 && stride < 0) || (startDiff > 0 && stride > 0)) &&
          //    startDiff < stride
          // but we don't have logical And or Or in the IR.

          const Expr* startDiff = IRSimplifier::simplify(
              new Sub(info->bounds[b].start, other->bounds[b].start));
          const Expr* stride = loopStrides[a][b];

          // TODO this logic assumes execution order is important but it may not
          // be valid with parallelism.

          const Expr* diffNeg = IRSimplifier::simplify(
              new CompareSelect(startDiff, new IntImm(0), kLT));
          const Expr* strideNeg = IRSimplifier::simplify(
              new CompareSelect(stride, new IntImm(0), kLT));

          std::cout << "safety? " << *startDiff << " " << *loopStrides[a][b]
                    << "\n";

          // not symbolic...
          if (diffNeg->isConstant() && strideNeg->isConstant()) {
            bool negative = immediateAs<int>(diffNeg)
                ? immediateAs<int>(strideNeg)
                : !immediateAs<int>(strideNeg);
            if (immediateAs<int>(diffNeg) == immediateAs<int>(strideNeg)) {
              CompareSelectOperation op = negative ? kLE : kGT;
              const Expr* check = IRSimplifier::simplify(
                  new CompareSelect(startDiff, loopStrides[a][b], op));
              std::cout << *check << "\n";
              if (check->isConstant() && immediateEquals<int>(check, 1)) {
                info->print();
                std::cout << " SAFE with ";
                other->print();
                std::cout << "\n";
                safe = true;
                break;
              }
            }
          }
        }

        if (!safe && overlaps(info->bounds, other->bounds) != NoOverlap) {
          // std::cout << "For handling adding dependency between: ";
          // info->print();
          // std::cout << " and ";
          // other->print();
          info->addDependency(other);
          other->addDependent(info);
          break;
        }
      }
    }

    mergeScope(currentScope_, currentScope_->parent);
    currentScope_ = currentScope_->parent;
  }

  void visit(const Block* v) override {
    auto prev_scope = currentScope_;
    if (currentScope_->block != v) {
      currentScope_ = std::make_shared<Scope>((Block*)v, prev_scope);
    }

    for (auto* s : *v) {
      s->accept(this);
    }

    for (auto* v : currentScope_->localVars) {
      knownVarBounds_.erase(v);
    }
    for (auto& pair : currentScope_->shadowedVarBounds) {
      knownVarBounds_[pair.first] = pair.second;
    }

    if (currentScope_ != prev_scope) {
      mergeScope(currentScope_, prev_scope);
      currentScope_ = prev_scope;
    }
  }

#define STMT_ON_STACK(Op)                    \
  virtual void visit(const Op* v) override { \
    const Stmt* last = lastStmt_;            \
    lastStmt_ = v;                           \
    IRVisitor::visit(v);                     \
    lastStmt_ = last;                        \
  }
  STMT_ON_STACK(AtomicAdd);
  STMT_ON_STACK(Allocate);
  STMT_ON_STACK(Free);
  STMT_ON_STACK(Let);
  STMT_ON_STACK(Cond);

#undef STMT_ON_STACK

  VarBoundMap knownVarBounds_;

  std::shared_ptr<Scope> currentScope_;

  void updateWriteHistory(
      std::deque<BoundRelationship>& writeHistory,
      std::shared_ptr<AccessInfo> info) {
    // std::cout << &writeHistory << "\n";
    // std::cout << "Update Write History (" << writeHistory.size() << ") for ";
    // info->print();

    bool isWrite = info->isWrite();

    for (auto it = writeHistory.rbegin(); it != writeHistory.rend();) {
      auto& indexBounds = it->first;
      auto& other = it->second;
      // std::cout << "writeHistory: ";
      // for (auto& r : indexBounds) {
      //   r.print();
      // }
      // std::cout << " => " << other->id << "\n";
      if (info->hasDependency(other)) {
        ++it;
        continue;
      }

      OverlapKind overlap = overlaps(indexBounds, info->bounds);

      if (overlap == NoOverlap) {
        ++it;
        continue;
      }

      // for (auto& r : indexBounds) {
      //   r.print();
      // }
      // std::cout << " somewhat overlaps ";
      // for (auto& r : info->bounds) {
      //   r.print();
      // }
      // std::cout << "\n";

      if (!isWrite) {
        // std::cout << "Adding dependencies between";
        // info->print();
        // std::cout << " and ";
        // other->print();
        info->addDependency(other);
        other->addDependent(info);
        ++it;
        continue;
      }

      if (overlap == TotalOverlapB) {
        // other->print();
        // std::cout << " ERASING due to to TotalOverlap with ";
        // info->print();
        // std::cout << "\n";

        // This is a reverse iterator, so advance it then erase the base to
        // erase the current element.
        std::advance(it, 1);
        writeHistory.erase(it.base());
      } else {
        auto newBounds =
            subtractIndicesBounds(indexBounds, info->bounds, overlap);

        std::advance(it, 1);
        bool breakOut = it == writeHistory.rend();
        writeHistory.erase(it.base());
        // std::cout << writeHistory.size() << "\n";

        // for (auto& p : writeHistory) {
        //   for (auto& b : p.first) {
        //     b.print();
        //   }
        //   std::cout << "\n";
        // }

        auto insertIt = it.base();

        for (auto& b : newBounds) {
          writeHistory.insert(insertIt, std::make_pair(b, other));
        }

        // std::cout << writeHistory.size() << " ";

        // for (auto& p : writeHistory) {
        //   for (auto& b : p.first) {
        //     b.print();
        //   }
        //   std::cout << "\n";
        // }

        if (breakOut) {
          break;
        }
      }
    }

    if (isWrite) {
      // std::cout << "inserting\n";
      writeHistory.push_back(std::make_pair(info->bounds, info));
    }
  }

  void mergeScope(std::shared_ptr<Scope> child, std::shared_ptr<Scope> parent) {
    // std::cout << "merge " << child->accesses_.size() << " "
    //           << parent->accesses_.size() << "\n";

    // for (auto& pair : parent->openWrites_) {
    //   std::cout << *pair.first << "\n";
    //   for (auto& p : pair.second) {
    //     for (auto& b : p.first) {
    //       std::cout << "\t";
    //       b.print();
    //     }
    //     std::cout << " - ";
    //     p.second->print();
    //   }
    // }

    // Update dependencies.
    for (auto& info : child->accesses_) {
      auto& writeHistory = parent->openWrites_[info->var];
      updateWriteHistory(writeHistory, info);
    }

    // for (auto& pair : parent->openWrites_) {
    //   std::cout << *pair.first << "\n";
    //   for (auto& p : pair.second) {
    //     for (auto& b : p.first) {
    //       std::cout << "\t";
    //       b.print();
    //     }
    //     std::cout << " - ";
    //     p.second->print();
    //   }
    // }
    parent->accesses_.insert(
        parent->accesses_.end(),
        std::make_move_iterator(child->accesses_.begin()),
        std::make_move_iterator(child->accesses_.end()));
    child->accesses_.clear();
  }

  class VarBoundBinder : public IRVisitor {
   public:
    VarBoundBinder(const VarBoundMap& vars) : vars_(vars) {}

    Bound getBounds(const Expr* e) {
      min_ = e;
      max_ = e;
      e->accept(this);
      min_ = IRSimplifier::simplify(min_);
      max_ = IRSimplifier::simplify(max_);
      return {min_, max_, new IntImm(0)};
    }

   private:
    void visit(const Var* v) {
      auto it = vars_.find(v);
      if (it == vars_.end()) {
        return;
      }

      min_ = Substitute(min_, {{v, it->second.start}});
      max_ = Substitute(max_, {{v, it->second.end}});
    }

    const Expr* min_{nullptr};
    const Expr* max_{nullptr};
    const VarBoundMap& vars_;
  };

  std::vector<Bound> getIndicesBounds(std::vector<const Expr*> indices) {
    std::vector<Bound> bounds;
    bounds.reserve(indices.size());
    VarBoundBinder binder(knownVarBounds_);
    for (auto* s : indices) {
      bounds.push_back(binder.getBounds(s));
    }
    return bounds;
  }

  size_t nextAccess_{0};
  const Stmt* lastStmt_{nullptr};
}; // namespace mem_dependency

int gcd(int a, int b) {
  if (b == 0)
    return a;
  else
    return gcd(b, a % b);
}

int lcm(int a, int b) {
  return a * b / gcd(a, b);
}

// Modifies the end of the bound to be on the correct stride.
// Returns the bound unchanged if the stride or size of the bound is not a known
// constant.
Bound normalizeBound(Bound a) {
  if (!a.stride->isConstant()) {
    return a;
  }

  int stride = immediateAs<int>(a.stride);

  // Default stride.
  if (stride == 0) {
    stride = 1;
    a.stride = new IntImm(1);
  }

  // Make negative bounds positive.
  if (stride < 0) {
    stride = 0 - stride;
    const Expr* newStride = new IntImm(stride);
    const Expr* newStart = a.end;

    // when flipping a strided bound we need to determine the start offset of
    // the new bound.
    if (stride > 1) {
      const Expr* startOffset =
          a.start->isConstant() ? new Mod(a.start, newStride) : a.start;
      const Expr* endOffset =
          a.end->isConstant() ? new Mod(a.end, newStride) : a.end;

      const Expr* offset = IRSimplifier::simplify(
          new Mod(new Sub(startOffset, endOffset), newStride));
      newStart = IRSimplifier::simplify(new Add(newStart, offset));
    }
    return normalizeBound({newStart, a.start, newStride});
  }

  const Expr* size = IRSimplifier::simplify(new Sub(a.end, a.start));
  if (!size->isConstant()) {
    return a;
  }

  // remove the unstrided results from the end.
  // end = end - (end - start) % stride;
  a.end = IRSimplifier::simplify(new Sub(a.end, new Mod(size, a.stride)));

  return a;
}

// returns PartialOverlap if any elements in the bound b are in the bound a.
// returns TotalOverlapA if all elements in the bound b are in the bound a.
// returns TotalOverlapB if all elements in the bound a are in the bound b, or
// if they are equal. returns NoOverlap if no elements in the bound b are in the
// bound a.
//
OverlapKind boundOverlap(Bound a, Bound b) {
  a = normalizeBound(a);
  b = normalizeBound(b);

  bool stridesEqual = exprEquals(a.stride, b.stride);
  // If they're equal they're equal.
  bool startEqual = exprEquals(a.start, b.start);
  bool endEqual = exprEquals(a.end, b.end);
  if (stridesEqual && startEqual && endEqual) {
    // std::cout << "YES! equal total\n";
    return TotalOverlapB;
  }

  bool stridesDistinct = false;
  bool strideIsOne = stridesEqual && a.stride->isConstant() &&
      (abs(immediateAs<int>(a.stride)) <= 1);

  // Whether the A stride includes all values, or the B stride. Can be both if
  // the strides are the same.
  bool ADominantStride = stridesEqual;
  bool BDominantStride = stridesEqual;

  const Expr* minStride = a.stride;
  const Expr* maxStride = a.stride;
  if (!stridesEqual) {
    minStride = IRSimplifier::simplify(new Min(a.stride, b.stride, true));
    maxStride = IRSimplifier::simplify(new Max(a.stride, b.stride, true));

    // std::cout << *minStride << " / " << *maxStride << "   ";

    if (minStride->isConstant()) {
      int mS = immediateAs<int>(minStride);
      if (immediateAs<int>(minStride) <= 1) {
        strideIsOne = true;

        if (exprEquals(minStride, a.stride)) {
          ADominantStride = true;
        } else {
          BDominantStride = true;
        }

      } else {
        const Expr* mod = IRSimplifier::simplify(new Mod(maxStride, minStride));
        // std::cout << *mod << " - ";
        if (mod->isConstant()) {
          int modI = immediateAs<int>(mod);
          if (modI == 1) {
            strideIsOne = true;
          } else if (modI > 1) {
            stridesDistinct = true;
          } else if (modI == 0) {
            if (exprEquals(minStride, a.stride)) {
              ADominantStride = true;
            } else {
              BDominantStride = true;
            }
          }
        }
      }
    }
  }

  const Expr* aSize = IRSimplifier::simplify(new Sub(a.end, a.start));
  const Expr* bSize = IRSimplifier::simplify(new Sub(b.end, b.start));

  bool constantSizes = aSize->isConstant() && bSize->isConstant();

  int aStride = a.stride->isConstant() ? abs(immediateAs<int>(a.stride)) : 0;
  int bStride = b.stride->isConstant() ? abs(immediateAs<int>(b.stride)) : 0;

  // If the size of a bound is 1 elem, then the stride doesn't matter.
  if (!stridesEqual && aSize->isConstant() && immediateAs<int>(aSize) == 0) {
    // std::cout << "BLERT";
    a.stride = b.stride;
    aStride = bStride;
    BDominantStride = true;
    stridesDistinct = false;
    if (bStride > 0) {
      strideIsOne = false;
    }
    minStride = b.stride;
  }

  if (!stridesEqual && bSize->isConstant() && immediateAs<int>(bSize) == 0) {
    b.stride = a.stride;
    // std::cout << "BORK";
    bStride = aStride;
    ADominantStride = true;
    stridesDistinct = false;
    if (aStride > 0) {
      strideIsOne = false;
    }
    minStride = a.stride;
  }
  if (ADominantStride && BDominantStride) {
    stridesEqual = true;
  }

  // std::cout << "stride(" << stridesEqual << ", " << stridesDistinct << ", "
  // << strideIsOne << ", " << ADominantStride << ", " << BDominantStride << ")
  // - "
  // << aStride << " " << bStride << " ";
  // std::cout << std::flush;

  if (!stridesDistinct && !strideIsOne) {
    const Expr* startDiff = IRSimplifier::simplify(new CompareSelect(
        a.start,
        b.start,
        new Sub(a.start, b.start),
        new Sub(b.start, a.start),
        CompareSelectOperation::kGT));
    // std::cout << *startDiff << " " << *minStride << "\n";
    if (startDiff->isConstant() && !immediateEquals(startDiff, 0)) {
      const Expr* check = IRSimplifier::simplify(new Mod(startDiff, minStride));
      // std::cout << "check " << *check << " ";
      if (check->isConstant() && immediateAs<int>(check) != 0) {
        // std::cout << "STRIDES ARE DISTINCT!!!\n";
        return NoOverlap;
      }
    }
  }

  const Expr* lowDiff = IRSimplifier::simplify(new Sub(a.start, b.end));
  const Expr* highDiff = IRSimplifier::simplify(new Sub(b.start, a.end));

  if (lowDiff->isConstant() && highDiff->isConstant()) {
    int low = immediateAs<int>(lowDiff);
    int high = immediateAs<int>(highDiff);
    // std::cout << low << " " << high;
    // No overlap.
    if (low > 0 || high > 0) {
      // std::cout << "NO! diffs positive\n";
      return NoOverlap;
    }
  }

  // std::cout << " " << *aSize << " " << *bSize << " ";

  // std::cout << "a:b " << aStride << ":" << bStride << " == ";
  if (highDiff->isConstant() && immediateAs<int>(highDiff) == 0) {
    if (BDominantStride && aSize->isConstant() &&
        immediateAs<int>(aSize) <= 0) {
      // std::cout << "Yes - total s1 high A\n";
      return TotalOverlapB;
    }

    if (ADominantStride && bSize->isConstant() &&
        immediateAs<int>(bSize) <= 0) {
      // std::cout << "Yes - total s1 high B\n";
      return TotalOverlapA;
    }
    // std::cout << "YES! highDiff 0\n";
    return PartialOverlap;
  }

  if (lowDiff->isConstant() && immediateAs<int>(lowDiff) == 0) {
    if (BDominantStride && aSize->isConstant() &&
        immediateAs<int>(aSize) <= 0) {
      // std::cout << "Yes - total s1 high A\n";
      return TotalOverlapB;
    }

    if (ADominantStride && bSize->isConstant() &&
        immediateAs<int>(bSize) <= 0) {
      // std::cout << "Yes - total s1 high B\n";
      return TotalOverlapA;
    }
    // std::cout << "YES! lowDiff 0\n";
    return PartialOverlap;
  }

  const Expr* diffs = IRSimplifier::simplify(new Sub(b.start, a.start));
  const Expr* diffe = IRSimplifier::simplify(new Sub(b.end, a.end));

  // If one side fully encloses the other, they're adjacent.
  if (!stridesDistinct && diffs->isConstant() && diffe->isConstant()) {
    int ds_i = immediateAs<int>(diffs);
    int de_i = immediateAs<int>(diffe);
    // std::cout << "DS: " << ds_i << " " << de_i << "\n";
    // If diffs and diffe have different signs they are enclosing.
    if (BDominantStride && ds_i < bStride && de_i > -bStride) {
      // std::cout << "YES! enclosed B\n";
      return TotalOverlapB;
    }

    if (ADominantStride && ds_i > -aStride && de_i < aStride) {
      // std::cout << "YES! enclosed A\n";
      return TotalOverlapA;
    }
  }

  if (constantSizes && aStride && bStride && !stridesEqual &&
      immediateAs<int>(maxStride) > 1) {
    int l = lcm(std::min(aStride, bStride), std::max(aStride, bStride));
    // std::cout << "LCM(" << aStride << " " << bStride << "): " << l << " ";
    int aSi = immediateAs<int>(aSize);
    int bSi = immediateAs<int>(bSize);

    if (aSi != bSi) {
      // std::cout << "size against LCM check " << aSi << " " << bSi << " " << l
      // << "\n";
      if (aSi >= l && bSi >= l) {
        // std::cout << "Yes partial due to LCM\n";
        return PartialOverlap;
      }

      int aS = immediateAs<int>(a.start);
      int bS = immediateAs<int>(b.start);
      int aE = immediateAs<int>(a.end);
      int bE = immediateAs<int>(b.end);

      int start = std::max(aS, bS);
      int end = std::min(aE, bE);
      int stride, check, offset;

      if (aStride > bStride) {
        stride = aStride;
        check = bStride;
        // start must be in the stride.
        if (start != aS) {
          int startOffset = aS % stride;
          int currOffset = start % stride;
          // std::cout << "HERE " << startOffset << " " << currOffset << " "
          // << stride << "\n";
          if (currOffset <= startOffset) {
            start += startOffset - currOffset;
          } else {
            start += (stride + startOffset) - currOffset;
          }
        }
        offset = bS % check;
      } else {
        stride = bStride;
        check = aStride;

        if (start != bS) {
          int startOffset = bS % stride;
          int currOffset = start % stride;
          if (currOffset <= startOffset) {
            start += startOffset - currOffset;
          } else {
            start += (stride + startOffset) - currOffset;
          }
        }
        offset = aS % check;
      }

      // std::cout << "\n YARGH " << start << " " << end << " " << stride << " "
      // << check << " " << offset << "\n";
      // std::cout << "Loop Size : " << ((end - start) / stride) << "\n";

      for (int i = start; i <= end; i += stride) {
        // std::cout << i << " " << offset << " " << check << "\n";
        if ((i + offset) % check == 0) {
          // std::cout << "Yes, loop : " << i << " " << check << "\n";
          return PartialOverlap;
        }
      }

      return NoOverlap;
    }
  }

  // std::cout << "Weak partial overlap\n";

  // We can't be sure there's no overlap so the conservative answer is
  // partial.
  return PartialOverlap;
}

std::vector<Bound> subtractBound(Bound a, Bound b, OverlapKind overlap) {
  bool constantStrides = a.stride->isConstant() && b.stride->isConstant();
  bool stridesEqual = exprEquals(a.stride, b.stride);

  a = normalizeBound(a);
  b = normalizeBound(b);

  // The bounds must overlap.
  std::vector<Bound> res;

  const Expr* lowDiff = IRSimplifier::simplify(new Sub(b.start, a.start));
  const Expr* highDiff = IRSimplifier::simplify(new Sub(b.end, a.end));

  // If the diff has only a single var, we can use the stride to try to work out
  // sign.
  if (!lowDiff->isConstant()) {
    auto vars = VarFinder::find(lowDiff);
    if (vars.size() == 1) {
      lowDiff = IRSimplifier::simplify(new Sub(
          Substitute(b.start, {{*vars.begin(), b.stride}}),
          Substitute(a.start, {{*vars.begin(), a.stride}})));
    }
  }

  if (!highDiff->isConstant()) {
    auto vars = VarFinder::find(highDiff);
    if (vars.size() == 1) {
      highDiff = IRSimplifier::simplify(new Sub(
          Substitute(b.end, {{*vars.begin(), b.stride}}),
          Substitute(a.end, {{*vars.begin(), a.stride}})));
    }
  }

  bool hasHead = lowDiff->isConstant() && immediateAs<int>(lowDiff) > 0;
  bool hasTail = highDiff->isConstant() && immediateAs<int>(highDiff) < 0;

  bool constantExtents = lowDiff->isConstant() && highDiff->isConstant();

  const Expr* varStart = a.start;
  const Expr* varEnd = a.end;

  if (!constantExtents) {
    // If we can't infer the bound lengths, there's no way to create a safe
    // subset. Just bail out.
    return {a};
  }

  if (!constantStrides && hasTail) {
    // There's no way to safely start the tail bound, since we don't know the
    // stride offset to use for an unknown stride. Just bail out.
    return {a};
  }

  if (hasHead) {
    // We need to shift the start position of the body forward to have the same
    // stride offset as the original a.
    //
    // start = b.start - b.start % a.stride + a.start % a.stride + a.stride
    //
    // We can use the quotient remainder theorem here to reorder, so dynamic
    // bounds can cancel out if necessary.
    //
    // start = b.start - (a.start + b.start % a.stride) + a.stride.

    varStart = IRSimplifier::simplify(new Add(
        new Sub(b.start, new Mod(new Add(a.start, b.start), a.stride)),
        a.stride));

    res.push_back({a.start,
                   IRSimplifier::simplify(new Sub(b.start, new IntImm(1))),
                   a.stride});
  }

  // initialize the integral strides to use.
  int aStride = constantStrides ? immediateAs<int>(a.stride) : 0;
  int bStride = constantStrides ? immediateAs<int>(b.stride) : 0;
  // if set, contains the t
  const Expr* variableStride = nullptr;

  if (!constantStrides && !stridesEqual) {
    if (a.stride->isConstant()) {
      // A constant a stride cannot interact with a variable b stride.
      return {a};
    }

    if (b.stride->isConstant() && immediateAs<int>(b.stride) <= 1) {
      bStride = 1;
      aStride = 1;
      variableStride = a.stride;
    } else {
      // both strides are non-constant.

      // If we can infer that the b.stride totally overlaps the a.stride then we
      // can handle that case.
      const Expr* modCheck =
          IRSimplifier::simplify(new Mod(a.stride, b.stride));
      if (modCheck->isConstant() && immediateEquals(modCheck, 0)) {
        return res;
      }

      // If we can infer that the b stride is bigger and a multiple of the a
      // stride we can continue.
      modCheck = IRSimplifier::simplify(new Mod(b.stride, a.stride));
      if (modCheck->isConstant() && immediateEquals(modCheck, 0)) {
        // if the modulus is 0, the div must be constant.
        bStride = immediateAs<int>(
            IRSimplifier::simplify(new Div(b.stride, a.stride)));
        aStride = 1;
        variableStride = a.stride;
      } else {
        // Can't infer, be conservative.
        return {a};
      }
    }
  }

  if (hasTail) {
    // For the end it doesn't matter as much as it will be fixed up by
    // normalizeBound.
    varEnd = b.end;
  }

  if (bStride > aStride || (aStride > bStride && aStride % bStride != 0)) {
    // this check above is about the b stride totally overlapping the A
    // stride. e.g. if A is stride 2 and B is stride 1 it necessarily includes
    // all elems in A within the body range.

    int minStride = std::min(aStride, bStride);
    int maxStride = std::max(aStride, bStride);

    int strideLCM = lcm(maxStride, minStride);
    int strideIters = strideLCM / aStride;

    // TODO: sensible threshold?????????
    if (strideIters > 20) {
      return {a};
    }

    // The end of the new bounds is either the end of a, the element before
    // the tail, or if that element is present in b, the element before that.
    const Expr* end = varEnd;
    const Expr* endInBcheck =
        IRSimplifier::simplify(new Mod(new Sub(end, b.start), b.stride));
    if (endInBcheck->isConstant() && immediateEquals(endInBcheck, 0)) {
      end = IRSimplifier::simplify(new Sub(varEnd, new IntImm(1)));
    }

    // OK we're going to iterate over the LCM by the stride of a.
    // The subtraction removes values when this is true:
    //    (varStart + iter) % b.stride == b.start % b.start
    // But, it's hard to compare modulus of dynamic bounds, so we can rewrite
    // as:
    //
    //    ((varStart + iter) % b.stride - b.start % b.stride ) % b.stride == 0
    //
    // Then via the quotient remainder theorem we can rewrite as:
    //
    //    ((varStart + iter) - b.start) % b.stride == 0
    //
    // which allows us to cancel out the dynamic part when possible.

    const Expr* stride = new IntImm(strideLCM);
    if (variableStride) {
      stride = IRSimplifier::simplify(new Mul(stride, variableStride));
    }

    for (int i = 0; i < strideIters; ++i) {
      const Expr* iter = new IntImm(i * aStride);
      if (variableStride) {
        iter = IRSimplifier::simplify(new Mul(iter, variableStride));
      }
      const Expr* diff = IRSimplifier::simplify(
          new Mod(new Sub(new Add(varStart, iter), b.start), b.stride));

      if (!diff->isConstant() || !immediateEquals(diff, 0)) {
        res.push_back(
            {IRSimplifier::simplify(new Add(varStart, new IntImm(i * aStride))),
             end,
             stride});
      }
    }
  }

  for (auto& r : res) {
    r = normalizeBound(r);
  }

  if (hasTail) {
    // The start of the tail is the next element after the end of B that is in
    // the stride of a.
    //
    // tail start = b.end - (b.end % a.stride + a.start % a.stride) + a.stride
    //
    // need to use the quotient remainder theorem here too to handle dynamic
    // bounds.
    //
    // tail start = b.end - (b.end + a.start) % a.stride + a.stride
    const Expr* offset =
        IRSimplifier::simplify(new Mod(new Add(b.end, a.start), a.stride));

    const Expr* tailStart =
        IRSimplifier::simplify(new Add(new Sub(b.end, offset), a.stride));

    Bound tail(tailStart, a.end, a.stride);
    tail = normalizeBound(tail);

    const Expr* size = IRSimplifier::simplify(new Sub(tail.end, tail.start));
    bool oneElem = size->isConstant() && immediateEquals(size, 0);
    bool merged = false;

    for (auto& r : res) {
      if (exprEquals(tail.stride, r.stride)) {
        auto* check = IRSimplifier::simplify(
            new Sub(new Sub(tail.start, r.end), tail.stride));
        if (check->isConstant() && immediateEquals(check, 0)) {
          r.end = tail.end;
          merged = true;
          break;
        }
      }

      if (oneElem) {
        auto* check = IRSimplifier::simplify(
            new Sub(new Sub(tail.start, r.end), r.stride));
        if (check->isConstant() && immediateEquals(check, 0)) {
          r.end = tail.end;
          merged = true;
          break;
        }
      }

      const Expr* rSize = IRSimplifier::simplify(new Sub(r.end, r.start));
      if (rSize->isConstant() && immediateEquals(rSize, 0)) {
        auto* check = IRSimplifier::simplify(
            new Sub(new Sub(r.start, tail.end), r.stride));

        if (check->isConstant() && immediateEquals(check, 0)) {
          r.end = tail.end;
          r.stride = tail.stride;
          merged = true;
          break;
        }
      }
    }

    if (!merged) {
      res.push_back(tail);
    }
  }

  return res;
}

std::vector<Bound> subtractBound(Bound a, Bound b) {
  OverlapKind overlap = boundOverlap(a, b);
  if (overlap == NoOverlap) {
    return {normalizeBound(a)};
  }
  if (overlap == TotalOverlapB) {
    return {};
  }

  return subtractBound(a, b, overlap);
}

} // namespace mem_dependency
} // namespace tensorexpr
} // namespace jit
} // namespace torch
