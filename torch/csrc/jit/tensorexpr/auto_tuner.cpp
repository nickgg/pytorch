
#include <torch/csrc/jit/tensorexpr/auto_tuner.h>

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"

namespace torch {
namespace jit {
namespace tensorexpr {

AutoTuner::~AutoTuner() {}

class VectorizeChecker : public IRVisitor {
 public:
  virtual void visit(const Ramp* v) {
    vectorized = true;
    IRVisitor::visit(v);
  }
  virtual void visit(const Broadcast* v) {
    vectorized = true;
    IRVisitor::visit(v);
  }

  static bool isVectorized(Stmt* s) {
    VectorizeChecker ch;
    s->accept(&ch);
    return ch.vectorized;
  }

  bool vectorized{false};
};

void AutoTuner::generateSplitWithTail(Candidate* c, int factor) {
  std::vector<Candidate*> new_candidates;

  auto initialLoops = NodeFinder<For>::find(c->loopnest.root_stmt());

  for (int l = 0; l < initialLoops.size(); ++l) {
    LoopNest temp(c->loopnest);
    For* target = NodeFinder<For>::find(temp.root_stmt())[l];

    For *o, *i, *t;
    temp.splitWithTail(target, factor, &o, &i, &t);

    Candidate* n = new Candidate(temp, c->schedule);
    n->schedule.push_back(
        new tuning::SplitWithTail(target->var()->name_hint(), factor));
    new_candidates.push_back(n);
  }

  for (auto* c : new_candidates) {
    generateVectorize(c);
    addPotentialCandidate(c);
  }
}

void AutoTuner::generateSplitWithMask(Candidate* c, int factor) {
  // OK so this is hacky.
  auto initialLoops = NodeFinder<For>::find(c->loopnest.root_stmt());
  for (int l = 0; l < initialLoops.size(); ++l) {
    LoopNest temp(c->loopnest);
    For* target = NodeFinder<For>::find(temp.root_stmt())[l];
    For *o, *i;
    temp.splitWithMask(target, factor, &o, &i);

    Candidate* n = new Candidate(temp, c->schedule);
    n->schedule.push_back(
        new tuning::SplitWithMask(target->var()->name_hint(), factor));
    addPotentialCandidate(n);
  }
}

void AutoTuner::generateVectorize(Candidate* c) {
  // OK so this is hacky.
  auto initialLoops = NodeFinder<For>::find(c->loopnest.root_stmt());
  for (int l = 0; l < initialLoops.size(); ++l) {
    For* loop = initialLoops[l];
    // std::cout << "want to vectorize " << loop->var()->name_hint() << ": ";
    if (!NodeFinder<For>::find(loop->body()).empty()) {
      // std::cout << "not inner loop\n";
      continue;
    }

    if (VectorizeChecker::isVectorized(loop->body())) {
      // std::cout << "already vectorized\n";
      continue;
    }

    if (!loop->stop()->isConstant()) {
      // std::cout << "non const loop stop (" << *loop->stop() << ")\n";
      continue;
    }

    auto reduces = NodeFinder<const ReduceOp>::find(loop->body());
    if (!reduces.empty()) {
      // std::cout << "contains reductions\n";
      continue;
    }

    const Expr* s =
        IRSimplifier::simplify(new Sub(loop->stop(), loop->start()));

    if (!s->isConstant()) {
      // std::cout << "non constant length : " << *s << "\n";
      continue;
    }

    int loop_len = immediateAs<int>(s);
    if (loop_len == 4 || loop_len == 8 || loop_len == 16 || loop_len == 32) {
      LoopNest temp(c->loopnest);
      For* target = NodeFinder<For>::find(temp.root_stmt())[l];
      // std::cout << "I AM VECTORIZING!\n";
      temp.vectorize(target);

      Candidate* n = new Candidate(temp, c->schedule);
      n->schedule.push_back(new tuning::Vectorize(target->var()->name_hint()));
      addPotentialCandidate(n);
    } else {
    }
  }
}

void AutoTuner::generateReorder(Candidate* c) {
  // OK so this is hacky.
  auto initialLoops = NodeFinder<For>::find(c->loopnest.root_stmt());
  for (int l = 0; l < initialLoops.size(); ++l) {
    For* loop = initialLoops[l];
    auto internal_loops = NodeFinder<For>::find(loop->body());
    if (internal_loops.empty()) {
      continue;
    }

    for (int il = 0; il < internal_loops.size(); ++il) {
      LoopNest temp(c->loopnest);
      auto new_loops = NodeFinder<For>::find(temp.root_stmt());
      For* outer = new_loops[l];
      For* inner = new_loops[l + 1 + il];
      temp.reorderAxis(outer, inner);

      Candidate* n = new Candidate(temp, c->schedule);
      n->schedule.push_back(new tuning::ReorderAxis(
          outer->var()->name_hint(), inner->var()->name_hint()));
      addPotentialCandidate(n);
    }
  }
}

void AutoTuner::generateInlining(Candidate* c) {
  auto intermediates = c->loopnest.getIntermediateBufs();

  for (int i = 0; i < intermediates.size(); ++i) {
    LoopNest temp(c->loopnest);
    const Buf* buf = temp.getIntermediateBufs()[i];
    try {
      temp.computeInline(buf);
    } catch (std::exception& e) {
      continue;
    }

    Candidate* n = new Candidate(temp, c->schedule);
    n->schedule.push_back(new tuning::Inline(buf->name_hint()));
    addPotentialCandidate(n);
  }
}

void AutoTuner::generateRfactor(Candidate* c) {
  auto reductions = NodeFinder<ReduceOp>::find(c->loopnest.root_stmt());
  for (const ReduceOp* op : reductions) {
    if (op->reduce_args().size() < 2) {
      continue;
    }

    for (const Var* var : op->reduce_args()) {
      LoopNest temp(c->loopnest);
      temp.rfactor(op, var);

      Candidate* n = new Candidate(temp, c->schedule);
      n->schedule.push_back(new tuning::Rfactor(
          op->accumulator()->name_hint(), var->name_hint()));
      addPotentialCandidate(n);
    }
  }
}

/*
std::vector<AutoTuner::Candidate*> AutoTuner::generateComputeAt(Candidate* c) {
  std::vector<Candidate*> new_candidates;

  auto loads = NodeFinder<Load>::find(c->loopnest.root_stmt());
  auto stores = NodeFinder<Store>::find(c->loopnest.root_stmt());
  std::map<const Var*, std::vector<Load*>> consumers;
  std::map<const Var*, std::vector<Store*>> producers;
  std::set<const Var*> seen_vars;

  for (auto* l : loads) {
    consumers[l->base_handle()].push_back(l);
    seen_vars.insert(l->base_handle());
  }

  for (auto* s : stores) {
    producers[s->base_handle()].push_back(s);
    seen_vars.insert(s->base_handle());
  }

  for (auto* var : seen_vars) {
    auto c = consumers[var];
    auto p = producers[var];
    if (c.size() > 0 && p.size() > 0) {
      std::cout << "candidates for " << var->name_hint() << "\n";
      std::cout << "consumers:\n";
      for (auto* cc : c) {
        std::cout << "\t" << *cc << "\n";
      }
      std::cout << "producers:\n";
      for (auto* pp : p) {
        std::cout << "\t" << *pp << "\n";
      }
    }
  }

  return new_candidates;
}*/

bool AutoTuner::addPotentialCandidate(Candidate* c) {
  generated_++;
  auto start = std::chrono::high_resolution_clock::now();
  c->hash = hashCandidateStmt(c->loopnest.root_stmt());

  if (candidatesByHash_.emplace(c->hash, c).second == false) {
    return false;
  }
  auto end = std::chrono::high_resolution_clock::now();

  hashing_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (potential_candidates_.size() <= c->depth()) {
    potential_candidates_.resize(c->depth() + 1);
  }

  // if (c->depth() == 1) {
  //   std::cout << "adding depth 1: ";
  //   c->logSchedule();
  //   std::cout << "\n";
  // }
  potential_candidates_[c->depth()].push_back(c);

  return true;
}

void AutoTuner::generateChildren(Candidate* c, bool first) {
  generateReorder(c);
  generateInlining(c);
  generateRfactor(c);

  // if (!first_) {
  std::vector<int> factors = {2, 3, 4, 8, 16, 32, 64};
  for (int f : factors) {
    generateSplitWithTail(c, f);
    // Cond unimplemented in LLVM backend
    // generateSplitWithMask(c, f);
  }
  // }
}

void AutoTuner::generateNextCandidates() {
  auto c_it = resolved_candidates_.begin();
  int done = 0;
  while (done < 3 && c_it != resolved_candidates_.end()) {
    Candidate* c = *c_it++;
    if (c->children_generated) {
      continue;
    }
    generateChildren(c, first_);
    done++;
  }
  std::random_shuffle(resolved_candidates_.begin(), resolved_candidates_.end());
  c_it = resolved_candidates_.begin();

  while (done < 5 && c_it != resolved_candidates_.end()) {
    Candidate* c = *c_it++;
    if (c->children_generated) {
      continue;
    }
    generateChildren(c, first_);
    done++;
  }
  first_ = false;
}

std::deque<AutoTuner::Candidate*> AutoTuner::pickCandidates(size_t num) {
  // uhhh lets start with an even split of depths?
  size_t depth = potential_candidates_.size();
  int picked = 0;
  std::deque<Candidate*> results;
  std::cout << "potential by depth: ";
  for (int i = 0; i < depth; ++i) {
    std::deque<Candidate*>& depthList = potential_candidates_[i];
    std::cout << depthList.size() << " ";
    std::random_shuffle(depthList.begin(), depthList.end());
    while (picked < (i * (num / depth)) && !depthList.empty()) {
      results.push_back(depthList.front());
      depthList.pop_front();
      picked++;
    }
  }
  std::cout << "\n";

  while (picked < num) {
    bool found = false;
    for (int i = 0; i < depth; ++i) {
      std::deque<Candidate*>& depthList = potential_candidates_[i];
      if (depthList.empty()) {
        continue;
      }
      results.push_back(depthList.front());
      depthList.pop_front();
      picked++;
      found = true;
    }
    if (!found) {
      break;
    }
  }

  return results;
}

void AutoTuner::run(int iterations) {
  if (argData_.empty()) {
    GenerateCallArgs();
  }

  size_t initial = 0;
  if (resolved_candidates_.empty()) {
    std::cout << *rootNest_.root_stmt() << "\n";
    resolved_candidates_.push_back(new Candidate(LoopNest(rootNest_)));
    runCandidate(resolved_candidates_.back());
    initial = resolved_candidates_.back()->time.count();
    std::cout << "Initial Kernel: " << initial << "us\n";
    potential_candidates_.resize(1);

    // Now we have reference results:
    for (auto& pair : outputs_) {
      memcpy(
          referenceOutputs_[pair.first].ptr, pair.second.ptr, pair.second.len);
    }
    referenceReady_ = true;
  }

  // Generate Candidates
  for (int run = 0; run < iterations; ++run) {
    // Generate some new candidates.
    auto start = std::chrono::high_resolution_clock::now();
    try {
      generateNextCandidates();
    } catch (std::exception& e) {
      std::cout << "Exception in generateNextCandidates : " << e.what() << "\n";
    }
    auto gen_end = std::chrono::high_resolution_clock::now();
    generation_time +=
        std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - start);

    // Pick some candidates.
    auto candidates = pickCandidates(50);
    std::cout << "got " << candidates.size() << " candidates\n";
    size_t numCandidates = candidates.size();

    // Run Candidates.
    for (auto* c : candidates) {
      // c->logSchedule();
      // std::cout << std::flush;
      RunResult r = runCandidate(c);
      std::string lastOpName = c->schedule.back()->name();
      switch (r) {
        case SUCCESS:
          runStats_.success++;
          resolved_candidates_.push_back(c);
          perOpStats_[lastOpName].success++;
          // std::cout << " SUCCESS " << c->time.count() << " us\n";
          break;
        case CODEGEN_FAIL:
          runStats_.codegen_fail++;
          perOpStats_[lastOpName].codegen_fail++;
          // std::cout << " CODEGEN_FAIL\n";
          break;
        case RUN_FAIL:
          runStats_.run_fail++;
          perOpStats_[lastOpName].run_fail++;
          // std::cout << " RUN_FAIL " << c->runs << "\n";
          break;
        case BAD_OUTPUTS:
          runStats_.bad_output++;
          perOpStats_[lastOpName].bad_output++;
          // c->logSchedule();
          // std::cout << std::flush;
          // std::cout << " BAD_OUTPUTS " << c->runs << "\n";
          break;
      }
    }
    sortCandidates();

    // Now get some more confidence.
    bool confident = true;
    while (!confident) {
      // std::cout << "building condfidence in timing...\n";
      bool optimist = true;
      for (int i = 0; i < 10; ++i) {
        if (resolved_candidates_.size() <= i) {
          continue;
        }

        if (resolved_candidates_[i]->runs < ((1 + run) * 2)) {
          runCandidate(resolved_candidates_[i]);
          optimist = false;
        }
      }

      if (optimist) {
        confident = true;
      }

      sortCandidates();
    }

    auto end = std::chrono::high_resolution_clock::now();
    Candidate* best = resolved_candidates_.front();
    size_t best_time = best->time.count();
    double speedup = (1. - (double)best_time / (double)initial) * 100;
    std::cout << "Run " << run << " " << numCandidates << " candidates in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end - start)
                     .count()
              << "ms : " << generated_ << " generated, " << tested_
              << " tested, " << resolved_candidates_.size() << " total.\n";
    std::cout << "best Time : " << best_time << "us - speedup: " << speedup;
    std::cout << " (ran " << best->runs << " times) - hash: " << best->hash._h;
    std::cout << "\nBest schedule: ";
    best->logSchedule();
    std::cout << "\n";
    Stmt* simplified = best->loopnest.root_stmt()
                           ->accept_mutator(&simplifier)
                           ->accept_mutator(&expander);

    std::cout << *simplified << "\n";

    generated_ = 0;
    tested_ = 0;
  }

  std::cout << generation_time.count() << "ms generating candidates, "
            << codegen_time.count() << " ms generating code\n";
  std::cout << hashing_time.count() << "ms hashing, " << simplify_time.count()
            << "ms simplifying\n";
  std::cout << running_time.count() << "ms running candidates, "
            << sorting_time.count() << "ms sorting candidates\n";
  std::cout << checking_time.count() << "ms checking outputs\n";

  std::cout << runStats_.success << " success, " << runStats_.codegen_fail
            << " fail at codegen\n";
  std::cout << runStats_.run_fail << " fail at runtime, "
            << runStats_.bad_output << " had bad outputs\n";

  std::vector<std::string> opNames = {"Vectorize",
                                      "SplitWithMask",
                                      "SplitWithTail",
                                      "ReorderAxis",
                                      "Rfactor",
                                      "Inline"};

  for (auto& name : opNames) {
    RunStats& opStats = perOpStats_[name];
    std::cout << name << ": " << opStats.success << " " << opStats.codegen_fail
              << " " << opStats.run_fail << " " << opStats.bad_output << "\n";
  }

  // for (int i = 0; i < 20; ++i) {
  //   Candidate* c = resolved_candidates_[i];
  //   std::cout << "Candidate " << i << " (" << c->time.count() << "us | "
  //             << c->runs << "): ";
  //   c->logSchedule();
  //   std::cout << " - hash: " << c->hash._h << "\n";
  // }

  // Candidate* best = resolved_candidates_.front();
  // Stmt* simplified = best->loopnest.root_stmt()
  //                        ->accept_mutator(&simplifier)
  //                        ->accept_mutator(&expander);

  // std::cout << *simplified << "\n";
} // namespace tensorexpr

LoopNest AutoTuner::getBestCandidate() {
  if (resolved_candidates_.empty()) {
    throw std::runtime_error("must run AutoTuner before getting candidates");
  }

  sortCandidates();
  return resolved_candidates_.front()->loopnest;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T*>::type getRandomArg(
    size_t len) {
  T* ret = new T[len];

  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO what's a sensible distribution here?
  std::uniform_int_distribution<T> dis(0, 1000);

  for (size_t i = 0; i < len; ++i) {
    ret[i] = dis(gen);
    // ret[i] = i;
  }

  return ret;
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T*>::type getRandomArg(
    size_t len) {
  T* ret = new T[len];

  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO what's a sensible distribution here?
  std::uniform_real_distribution<T> dis(-1.0, 1.0);

  for (int i = 0; i < len; ++i) {
    ret[i] = dis(gen);
  }

  return ret;
}

void AutoTuner::GenerateCallArgs() {
  auto boundsInfo = inferBounds(rootNest_.root_stmt());

  std::set<const Buf*> output_bufs;
  for (auto* tensor : rootNest_.getOutputTensors()) {
    output_bufs.insert(tensor->buf());
  }

  for (auto b : boundsInfo) {
    const Expr* size = new IntImm(1);
    for (int i = 0; i < b.start.size(); ++i) {
      const Expr* dim = IRSimplifier::simplify(
          new Add(new IntImm(1), new Sub(b.stop[i], b.start[i])));
      size = new Mul(size, dim);
    }

    size = IRSimplifier::simplify(size);
    size_t s = immediateAs<int>(size);

    switch (b.buf->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                                  \
  case ScalarType::Name: {                                                     \
    const Var* v = b.buf->base_handle();                                       \
    argData_[v].emplace_back((void*)getRandomArg<Type>(s));                    \
    if (output_bufs.count(b.buf)) {                                            \
      outputs_[v] = {(void*)getRandomArg<Type>(s), s * sizeof(Type)};          \
      referenceOutputs_[v] = {(void*)getRandomArg<Type>(s), s * sizeof(Type)}; \
    }                                                                          \
  } break;
      AT_FORALL_SCALAR_TYPES(TYPE_CASE);
#undef TYPE_CASE
      case ScalarType::Half:
      case ScalarType::Bool:
      default:
        throw std::runtime_error("unsupported type in AutoTuner");
    }
  }

  for (auto& b : args_) {
    if (argData_.find(b.var()) == argData_.end()) {
      // arg doesn't actually exist but run will need it.
      std::cout << "inventing dummy arg for unused argument "
                << b.var()->name_hint() << "\n";
      argData_[b.var()].emplace_back((void*)getRandomArg<int>(1));
    }
  }
}

SimplifierHashType AutoTuner::hashCandidateStmt(Stmt* s) {
  Stmt* simplified = s->accept_mutator(&simplifier)->accept_mutator(&expander);
  return simplifier.hasher().hash(simplified);
}

void AutoTuner::sortCandidates() {
  auto start = std::chrono::high_resolution_clock::now();
  std::sort(
      resolved_candidates_.begin(),
      resolved_candidates_.end(),
      [](const Candidate* a, const Candidate* b) -> bool {
        return a->time < b->time;
      });
  auto end = std::chrono::high_resolution_clock::now();
  sorting_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

bool AutoTuner::checkOutputs() {
  auto start = std::chrono::high_resolution_clock::now();
  for (auto& pair : outputs_) {
    int* output = (int*)pair.second.ptr;
    int* reference = (int*)referenceOutputs_[pair.first].ptr;
    for (int i = 0; i < pair.second.len / 4; ++i) {
      if (output[i] != reference[i]) {
        auto end = std::chrono::high_resolution_clock::now();
        sorting_time +=
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return false;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  sorting_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return true;
}

AutoTuner::RunResult AutoTuner::runCandidate(Candidate* c) {
  // another clone because we want the candidate too stay
  // un-preparedForCodegen.
  LoopNest temp(c->loopnest);
  Stmt* s = nullptr;

  if (!c->codegen) {
    auto pre_simplify = std::chrono::high_resolution_clock::now();
    try {
      temp.prepareForCodegen();
    } catch (std::exception& e) {
      return CODEGEN_FAIL;
    }
    s = IRSimplifier::simplify(temp.root_stmt());
    auto pre_codegen = std::chrono::high_resolution_clock::now();
    try {
      c->codegen = std::make_unique<LLVMCodeGen>(s, args_);
    } catch (std::exception& e) {
      // std::cout << "Codegen Failed: " << *c->loopnest.root_stmt() << "\n";
      return CODEGEN_FAIL;
    }

    auto post_codegen = std::chrono::high_resolution_clock::now();
    // std::chrono::milliseconds cgtime =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(
    //         post_codegen - pre_codegen);
    // std::cout << "codegen time: " << cgtime.count() << "\n";
    // if (cgtime.count() >= 10000) {
    //   std::cout << "SLOW CODEGEN: " << *s << "\n";
    // }

    simplify_time += std::chrono::duration_cast<std::chrono::milliseconds>(
        pre_codegen - pre_simplify);
    codegen_time += std::chrono::duration_cast<std::chrono::milliseconds>(
        post_codegen - pre_codegen);
  }

  std::vector<CodeGen::CallArg> runArgs;
  for (auto& b : args_) {
    auto it = argData_.find(b.var());
    if (it != argData_.end()) {
      auto oit = outputs_.find(b.var());
      if (oit != outputs_.end()) {
        runArgs.push_back(oit->second.ptr);
      } else {
        runArgs.push_back(it->second.front());
      }
    } else {
      std::cout << "COULDNT FIND " << b.var()->name_hint() << "\n";
    }
  }

  tested_++;
  std::chrono::microseconds total(0);

  int new_runs = 5;

  // if (!c->schedule.empty()) {
  //   if (!s) {
  //     std::cout << "AAAA no s\n";
  //   }
  //   if (tuning::Rfactor* f =
  //           dynamic_cast<tuning::Rfactor*>(c->schedule.back())) {
  //     std::cout << *c->loopnest.root_stmt() << "\n";
  //   }
  // }

  try {
    for (int i = 0; i < new_runs; ++i) {
      for (auto& pair : outputs_) {
        int* output = (int*)pair.second.ptr;
        assert(argData_.find(pair.first) != argData_.end());
        CodeGen::CallArg& arg = argData_[pair.first].back();
        void* reference = arg.data();
        memcpy(output, reference, pair.second.len);
      }

      auto start = std::chrono::high_resolution_clock::now();
      c->codegen->call(runArgs);
      auto end = std::chrono::high_resolution_clock::now();
      total +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      if (referenceReady_ && !checkOutputs()) {
        c->runs += i + 1;
        return BAD_OUTPUTS;
      }
    }

  } catch (std::exception& e) {
    return RUN_FAIL;
  }
  running_time += std::chrono::duration_cast<std::chrono::milliseconds>(total);

  total += c->time * c->runs;
  c->runs += new_runs;
  c->time = total / c->runs;
  return SUCCESS;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
