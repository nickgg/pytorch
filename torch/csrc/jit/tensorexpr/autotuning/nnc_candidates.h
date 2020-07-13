#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/autotune_base.h>
#include <torch/csrc/jit/tensorexpr/autotuning/nnc_transforms.h>

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

template <class CandidateType>
class NNCTransformer {
 public:
  virtual CandidateType* mutateCandidate(
      std::vector<CandidateType*> children,
      CandidateType* c,
      typename CandidateType::Program* p,
      Transform* t) {
    auto* newc = new CandidateType(p, c, t);
    children.push_back(newc);
    return newc;
  }

  std::vector<CandidateType*> generateSplitWithTail(
      CandidateType* c,
      int factor) {
    std::vector<CandidateType*> children;
    auto initialLoops = NodeFinder<For>::find(c->program->root_stmt());

    for (int l = 0; l < initialLoops.size(); ++l) {
      LoopNest* temp = new LoopNest(*c->program);
      For* target = NodeFinder<For>::find(temp->root_stmt())[l];

      For *o, *i, *t;
      temp->splitWithTail(target, factor, &o, &i, &t);

      mutateCandidate(
          children,
          c,
          temp,
          new transforms::SplitWithTail(target->var()->name_hint(), factor));
    }
    return children;
  }

  CandidateType* cloneWithSplitWithMask(
      std::vector<CandidateType*> children,
      CandidateType* c,
      For* target,
      int factor) {
    if (factor < 2) {
      return nullptr;
    }

    LoopNest* temp = new LoopNest(*c->program);
    for (auto* f : NodeFinder<For>::find(temp->root_stmt())) {
      if (f->var() == target->var()) {
        For *o, *i;
        temp->splitWithMask(f, factor, &o, &i);

        return mutateCandidate(
            children,
            c,
            temp,
            new transforms::SplitWithMask(f->var()->name_hint(), factor));
      }
    }
    return nullptr;
  }

  std::vector<CandidateType*> generateSplitWithMask(
      CandidateType* c,
      int factor) {
    std::vector<CandidateType*> children;
    auto initialLoops = NodeFinder<For>::find(c->program->root_stmt());
    for (int l = 0; l < initialLoops.size(); ++l) {
      cloneWithSplitWithMask(children, c, initialLoops[l], factor);
    }
    return children;
  }

  std::vector<CandidateType*> mutateSplitWithMask(CandidateType* c) {
    std::vector<CandidateType*> children;
    if (c->schedule.transforms.empty()) {
      return children;
    }

    transforms::SplitWithMask* split =
        dynamic_cast<transforms::SplitWithMask*>(c->schedule.transforms.back());
    if (!split) {
      return children;
    }

    For* target = nullptr;
    for (auto* f : NodeFinder<For>::find(c->parent->program->root_stmt())) {
      // TODO this isn't uniqued
      if (f->var()->name_hint() == split->loopVar) {
        target = f;
        break;
      }
    }

    if (target != nullptr) {
      auto* thing = cloneWithSplitWithMask(
          children, (CandidateType*)c->parent, target, split->factor + 1);
      cloneWithSplitWithMask(
          children, (CandidateType*)c->parent, target, split->factor - 1);
      cloneWithSplitWithMask(
          children, (CandidateType*)c->parent, target, split->factor / 2);
      cloneWithSplitWithMask(
          children, (CandidateType*)c->parent, target, split->factor * 2);
    }
    return children;
  }

  std::vector<CandidateType*> generateReorderAxis(CandidateType* c) {
    std::vector<CandidateType*> children;
    auto initialLoops = NodeFinder<For>::find(c->program->root_stmt());

    for (int l = 0; l < initialLoops.size(); ++l) {
      For* loop = initialLoops[l];
      auto internal_loops = NodeFinder<For>::find(loop->body());
      if (internal_loops.empty()) {
        continue;
      }

      for (unsigned int il = 0; il < internal_loops.size(); ++il) {
        LoopNest* temp = new LoopNest(*c->program);
        auto new_loops = NodeFinder<For>::find(temp->root_stmt());
        For* outer = new_loops[l];
        For* inner = new_loops[l + 1 + il];
        temp->reorderAxis(outer, inner);

        mutateCandidate(
            children,
            c,
            temp,
            new transforms::ReorderAxis(
                outer->var()->name_hint(), inner->var()->name_hint()));
      }
    }
    return children;
  }

  std::vector<CandidateType*> generateInline(CandidateType* c) {
    std::vector<CandidateType*> children;
    /* TODO: need immediate inline PR.
    auto intermediates = c->loopnest.getIntermediateBufs();

    for (int i = 0; i < intermediates.size(); ++i) {
      LoopNest* temp = new LoopNest(c->loopnest);
      const Buf* buf = temp->getIntermediateBufs()[i];
      try {
        temp->computeInline(buf);
      } catch (std::exception& e) {
        continue;
      }

      mutateCandidate(temp, c, new transforms::Inline(buf->name_hint()));
    }*/
    return children;
  }

  std::vector<CandidateType*> generateRfactor(CandidateType* c) {
    std::vector<CandidateType*> children;
    auto reductions = NodeFinder<ReduceOp>::find(c->program->root_stmt());
    for (const ReduceOp* op : reductions) {
      if (op->reduce_args().size() < 2) {
        continue;
      }

      for (const Var* var : op->reduce_args()) {
        LoopNest* temp = new LoopNest(*c->program);
        temp->rfactor(op, var);

        mutateCandidate(
            children,
            c,
            temp,
            new transforms::Rfactor(
                op->accumulator()->name_hint(), var->name_hint()));
      }
    }
    return children;
  }

  std::vector<CandidateType*> generateNextAxisBinding(CandidateType* c) {
    std::vector<CandidateType*> children;
    auto initialLoops = NodeFinder<For>::find(c->program->root_stmt());
    for (int l = 0; l < initialLoops.size(); ++l) {
      if (c->nextBlockIdx < 3) {
        LoopNest* temp = new LoopNest(*c->program);
        For* target = NodeFinder<For>::find(temp->root_stmt())[l];
        if (target->loop_options().isDefault()) {
          temp->setGPUBlockIndex(target, c->nextBlockIdx);

          auto* n = mutateCandidate(
              children,
              c,
              temp,
              new transforms::BindBlockIdx(
                  target->var()->name_hint(), c->nextBlockIdx));
          if (!n) {
            continue;
          }
          n->nextBlockIdx++;
        }
      }

      if (c->nextThreadIdx < 3) {
        LoopNest* temp = new LoopNest(*c->program);
        For* target = NodeFinder<For>::find(temp->root_stmt())[l];
        if (target->loop_options().isDefault()) {
          temp->setGPUThreadIndex(target, c->nextThreadIdx);

          auto* n = mutateCandidate(
              children,
              c,
              temp,
              new transforms::BindThreadIdx(
                  target->var()->name_hint(), c->nextThreadIdx));
          if (!n) {
            continue;
          }
          n->nextThreadIdx++;
        }
      }
    }

    return children;
  }

  std::vector<CandidateType*> mutateAxisBinding(CandidateType* c) {
    std::vector<CandidateType*> children;
    auto initialLoops = NodeFinder<For>::find(c->program->root_stmt());
    for (int l = 0; l < initialLoops.size(); ++l) {
      if (!initialLoops[l]->loop_options().isDefault()) {
        for (int k = 0; k < initialLoops.size(); ++k) {
          if (l == k) {
            continue;
          }

          // Do a swap.
          LoopNest* temp = new LoopNest(*c->program);
          auto tempLoops = NodeFinder<For>::find(temp->root_stmt());
          For* before = tempLoops[l];
          For* after = tempLoops[k];

          // swap.
          auto tempOpts = before->loop_options();
          before->set_loop_options(after->loop_options());
          after->set_loop_options(tempOpts);

          mutateCandidate(
              children,
              c,
              temp,
              new transforms::SwapAxisIdx(
                  before->var()->name_hint(), after->var()->name_hint()));
        }
      }
    }
    return children;
  }
};

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch

