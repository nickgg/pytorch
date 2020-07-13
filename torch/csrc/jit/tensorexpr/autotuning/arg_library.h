#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/autotune_base.h>

#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

struct SizedBuffer {
  SizedBuffer(void* p, size_t l) : ptr(p), len(l) {}

  void* ptr{nullptr};
  size_t len{0};
};

class ArgumentLibrary {
 public:
  ArgumentLibrary(
      LoopNest* program,
      std::vector<CodeGen::BufferArg> args,
      size_t sets = 3)
      : args_(std::move(args)) {
    // Set up data container for each arg.
    for (auto& bA : args_) {
      argData_[bA.var()] = std::vector<CodeGen::CallArg>();
    }

    // Record which ones are outputs.
    for (auto* tensor : program->getOutputTensors()) {
      outputs_.insert(tensor->func_var()->base_handle());
    }

    // Find bounds and record sizes of each arg.
    auto boundsInfo = inferBounds(program->root_stmt());
    for (auto b : boundsInfo) {
      const Expr* size = new IntImm(1);
      for (unsigned int i = 0; i < b.start.size(); ++i) {
        const Expr* dim =
            new Add(new IntImm(1), new Sub(b.stop[i], b.start[i]));
        size = new Mul(size, dim);
      }

      size = IRSimplifier::simplify(size);
      assert(size->isConstant());

      argSizes_.emplace(
          b.buf->base_handle(),
          std::make_pair<Dtype, size_t>(
              b.buf->dtype(), immediateAs<int>(size)));
    }

    // It's possible that args contains buffers that are not necessary and not
    // present in the inferred bounds. CodeGen still expects us to provide them.
    for (auto& bA : args_) {
      if (argSizes_.count(bA.var()) == 0) {
        argSizes_.emplace(
            bA.var(), std::make_pair<Dtype, size_t>(ToDtype<int>(), 1));
      }
    }

    // Initialize random data.
    for (unsigned int i = 0; i < sets; ++i) {
      initializeArgset();
    }
  }

  const std::vector<CodeGen::BufferArg>& args() const {
    return args_;
  }

  size_t numArgsets() const {
    return numArgsets_;
  }

  size_t getArgBytes(const Var* v) const {
    auto it = argSizes_.find(v);
    if (it == argSizes_.end()) {
      return 0;
    }
    // it -> (dtype, size)
    return it->second.first.byte_size() * it->second.second;
  }

  bool isOutput(const Var* v) const {
    return outputs_.count(v) != 0;
  }

  size_t initializeArgset() {
    for (auto& p : argSizes_) {
      Dtype& dtype = p.second.first;
      size_t argSize = p.second.second;
      switch (dtype.scalar_type()) {
#define TYPE_CASE(Type, Name)                                               \
  case ScalarType::Name: {                                                  \
    SizedBuffer buf = getRandomArg<Type>(argSize);                          \
    argData_[p.first].emplace_back(buf.ptr);                                \
    if (outputs_.count(p.first)) {                                          \
      referenceOutputs_[p.first].emplace_back(getRandomArg<Type>(argSize)); \
    }                                                                       \
  } break;
        AT_FORALL_SCALAR_TYPES(TYPE_CASE);
#undef TYPE_CASE
        case ScalarType::Half:
        case ScalarType::Bool:
        default:
          throw std::runtime_error("unsupported type in AutoTuner");
      }
    }
    return ++numArgsets_;
  }

  std::vector<CodeGen::CallArg> getCallArgs(size_t idx) {
    std::vector<CodeGen::CallArg> runArgs;
    assert(numArgsets_ > 0);
    for (auto& b : args_) {
      auto it = argData_.find(b.var());
      assert(it != argData_.end());
      runArgs.push_back(it->second[idx % numArgsets_]);
    }

    return runArgs;
  }

  void setReferenceArgs(size_t idx, std::vector<CodeGen::CallArg> resolved) {
    referenceSets_++;
    assert(args_.size() == resolved.size());
    for (unsigned int i = 0; i < args_.size(); ++i) {
      const Var* var = args_[i].var();
      if (referenceOutputs_.count(var) == 0) {
        continue;
      }
      SizedBuffer& reference = referenceOutputs_[var][idx % numArgsets_];
      memcpy(reference.ptr, resolved[i].data(), reference.len);
    }
  }

  bool checkOutputs(size_t idx, std::vector<CodeGen::CallArg> resolved) {
    if (idx >= referenceSets_) {
      return true;
    }
    assert(args_.size() == resolved.size());
    for (unsigned int i = 0; i < args_.size(); ++i) {
      const Var* var = args_[i].var();
      if (referenceOutputs_.count(var) == 0) {
        continue;
      }

      SizedBuffer& reference = referenceOutputs_[var][idx % numArgsets_];
      switch (args_[i].dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)            \
  case ScalarType::Name: {               \
    if (!verifyArg(                      \
            (Type*)reference.ptr,        \
            (Type*)(resolved[i].data()), \
            reference.len)) {            \
      return false;                      \
    }                                    \
  } break;
        AT_FORALL_SCALAR_TYPES(TYPE_CASE);
#undef TYPE_CASE
        case ScalarType::Half:
        case ScalarType::Bool:
        default:
          throw std::runtime_error("unsupported type in AutoTuner - VA");
      }
    }

    return true;
  }

 protected:
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, SizedBuffer>::type
  getRandomArg(size_t len) {
    T* ret = new T[len];

    std::random_device rd;
    std::mt19937 gen(rd());
    // TODO what's a sensible distribution here?
    // very easy to overflow with random data
    std::uniform_int_distribution<T> dis(0, 10000);

    for (unsigned int i = 0; i < len; ++i) {
      ret[i] = dis(gen);
    }

    return SizedBuffer(ret, len * sizeof(T));
  }

  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value, SizedBuffer>::type
  getRandomArg(size_t len) {
    T* ret = new T[len];

    std::random_device rd;
    std::mt19937 gen(rd());
    // TODO what's a sensible distribution here?
    std::uniform_real_distribution<T> dis(-1.0, 1.0);

    for (unsigned int i = 0; i < len; ++i) {
      ret[i] = dis(gen);
    }

    return SizedBuffer(ret, len * sizeof(T));
  }

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, bool>::type verifyArg(
      T* A,
      T* B,
      size_t bytes) {
    size_t items = bytes / sizeof(T);
    for (unsigned int i = 0; i < items; ++i) {
      if (A[i] != B[i]) {
        return false;
      }
    }
    return true;
  }

  float FLOATING_POINT_EASE = 0.001;

  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value, bool>::type
  verifyArg(T* A, T* B, size_t bytes) {
    size_t items = bytes / sizeof(T);
    for (unsigned int i = 0; i < items; ++i) {
      if (std::fabs(A[i] - B[i]) > FLOATING_POINT_EASE) {
        return false;
      }
    }
    return true;
  }

  size_t numArgsets_{0};

  const std::vector<CodeGen::BufferArg> args_;

  std::unordered_map<const Var*, std::vector<CodeGen::CallArg>> argData_;
  std::unordered_map<const Var*, std::pair<Dtype, size_t>> argSizes_;

  std::set<const Var*> outputs_;
  std::unordered_map<const Var*, std::vector<SizedBuffer>> referenceOutputs_;
  size_t referenceSets_{0};
}; // namespace tuning

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch
