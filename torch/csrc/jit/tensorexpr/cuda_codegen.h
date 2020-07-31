#pragma once

#include <unordered_map>
#include <unordered_set>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// A class that analyzes the given program relevant for Cuda backends.
class CudaAnalysis : public IRVisitor {
 public:
  bool is_buf_store_target(const Buf* buf) const {
    return store_targets_.count(buf) > 0;
  }

  const Block* get_shared_scope(const Buf* buf) const {
    auto it = store_targets_.find(buf);
    if (it == store_targets_.end()) {
      return nullptr;
    }

    return it->second;
  }

 private:
  void visit(const Block* v) override {
    const Block* prev = last_scope_;
    last_scope_ = v;

    IRVisitor::visit(v);

    last_scope_ = prev;
  }

  void visit(const Store* v) override {
    auto it = store_targets_.find(v->buf());
    if (it == store_targets_.end()) {
      store_targets_[v->buf()] = last_scope_;
      return;
    }

    const Block* scope_ptr = last_scope_;
    std::unordered_set<const Block*> enclosing;
    while (scope_ptr != nullptr) {
      enclosing.insert(scope_ptr);
      Stmt* parent = scope_ptr->get_parent();
      if (!parent) {
        break;
      }
      while (parent != nullptr) {
        scope_ptr = dynamic_cast<const Block*>(parent);
        if (scope_ptr != nullptr) {
          break;
        }
        parent = parent->get_parent();
      }
    }

    const Block* prev = it->second;

    while (prev && enclosing.count(prev) == 0) {
      Stmt* parent = prev->get_parent();
      while (parent && !dynamic_cast<Block*>(parent)) {
        parent = parent->get_parent();
      }
      prev = dynamic_cast<Block*>(parent);
    }

    store_targets_[v->buf()] = prev;
  }

  std::unordered_map<const Buf*, const Block*> store_targets_;
  const Block* last_scope_{nullptr};
};

class GPUMetaVarRewriter : public IRMutator {
 public:
  explicit GPUMetaVarRewriter() {
    gpu_block_vars_ = {new Var("blockIdx.x", kInt),
                       new Var("blockIdx.y", kInt),
                       new Var("blockIdx.z", kInt)};
    gpu_thread_vars_ = {new Var("threadIdx.x", kInt),
                        new Var("threadIdx.y", kInt),
                        new Var("threadIdx.z", kInt)};
  }

  Stmt* mutate(const For* v) override;

  const std::vector<const Expr*>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  const std::vector<const Expr*>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

 private:
  std::vector<const Expr*> gpu_block_extents_;
  std::vector<const Expr*> gpu_thread_extents_;

  std::vector<const Var*> gpu_block_vars_;
  std::vector<const Var*> gpu_thread_vars_;
};

bool exprEquals(const Expr* A, const Expr* B) {
  return immediateEquals(IRSimplifier::simplify(new Sub(B, A)), 0);
}

class RegisterAliasMutator : public IRMutator {
 public:
  RegisterAliasMutator(
      const Buf* b,
      const std::vector<const Expr*>& i,
      const Var* v)
      : buf(b), indices(i), var(v) {
    varWrapper = new Buf(var, {}, b->dtype());
  }

  bool complete() {
    return bufFound > 0 && bufReplaced == bufFound;
  }

 private:
  Stmt* mutate(const Store* v) override {
    if (v->buf() == buf && indices.size() == v->indices().size()) {
      bufFound++;
      for (int i = 0; i < indices.size(); ++i) {
        if (!exprEquals(indices[i], v->indices()[i])) {
          return IRMutator::mutate(v);
        }
      }

      bufReplaced++;
      const Expr* value_new = v->value()->accept_mutator(this);
      return new Store(varWrapper, {}, value_new, new IntImm(1));
    }

    return IRMutator::mutate(v);
  }

  const Expr* mutate(const Load* v) override {
    if (v->buf() == buf && indices.size() == v->indices().size()) {
      bufFound++;
      for (int i = 0; i < indices.size(); ++i) {
        if (!exprEquals(indices[i], v->indices()[i])) {
          return v;
        }
      }

      bufReplaced++;
      return new Load(v->dtype(), varWrapper, {}, new IntImm(1));
    }

    return v;
  };

  const Buf* buf;
  std::vector<const Expr*> indices;
  const Var* var;
  const Buf* varWrapper;
  size_t bufFound{0};
  size_t bufReplaced{0};
};

// A class that overrides the underlying IRPrinter to produce Cuda C.
class CudaPrinter : public IRPrinter {
 public:
  explicit CudaPrinter(
      std::ostream* os,
      const CudaAnalysis* cuda_analysis,
      bool has_random)
      : IRPrinter(*os), cuda_analysis_(cuda_analysis) {
    if (has_random) {
      rand_func_ = new Var("rand", kHandle);
    }
  }

  void visit(const Cast* v) override;
  void visit(const Intrinsics* v) override;
  void visit(const For* v) override;

  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const AtomicAdd* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const Block* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;

  const Var* rand_func() const {
    return rand_func_;
  }

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  void maybe_insert_sync();
  const Var* rand_func_;
  const CudaAnalysis* cuda_analysis_;
  bool need_sync_ = false;
  std::unordered_set<const Var*> thread_local_bufs_;
};

// Construct Cuda C from the buffer and tensor input, and invoke the kernel
// when real arguments are provided.
class TORCH_CUDA_API CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  CudaCodeGen(Stmt* stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCUDA, at::cuda::current_device())) {
    Initialize();
  }

  CudaCodeGen(
      Stmt* stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::Device(at::kCUDA, at::cuda::current_device()))
      : CodeGen(stmt, buffer_args, device) {
    Initialize();
  }

  ~CudaCodeGen() override;

  void call(const std::vector<CallArg>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

 private:
  void Initialize();

  void CompileToNVRTC(const std::string& code, const std::string& func_name);

  UniqueNameManager* name_manager() {
    if (!printer_) {
      throw std::runtime_error("Null IRPrinter is not expected");
    }
    return printer_->name_manager();
  }

  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_;
  std::unique_ptr<CudaPrinter> printer_;
  std::unique_ptr<CudaAnalysis> cuda_analysis_;
  std::unique_ptr<GPUMetaVarRewriter> metavar_rewriter_;
  CUfunction function_;
  bool has_random_ = false;

  std::string GetUniqueFuncName(const std::string& func_prefix);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
