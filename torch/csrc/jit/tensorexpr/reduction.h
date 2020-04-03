#pragma once

#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/dim_arg.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/types.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using ParameterList = const std::vector<VarHandle>;
using ReduceInteraction = std::function<ExprHandle(ExprHandle, ExprHandle)>;
using ReduceBodyFunc = std::function<ExprHandle(ParameterList&)>;

class Reducer : public ExprNode<Reducer> {
 public:
  Reducer(
      ExprHandle accum,
      Stmt* init,
      ExprHandle body,
      ReduceInteraction c,
      const std::vector<const Var*>& reduce_args)
      : ExprNodeBase(body.dtype()),
        accumulator_(accum),
        initializer_(init),
        body_(body),
        interaction_(c),
        reduce_args_(reduce_args) {}

  ExprHandle accumulator() const {
    return accumulator_;
  }

  Stmt* initializer() const {
    return initializer_;
  }
  ExprHandle body() const {
    return body_;
  }

  ReduceInteraction interaction() const {
    return interaction_;
  }

  const std::vector<const Var*>& reduce_args() const {
    return reduce_args_;
  }

  ExprHandle complete() const {
    return interaction_(accumulator_, body_);
  }

 private:
  ExprHandle accumulator_;
  Stmt* initializer_;
  ExprHandle body_;
  ReduceInteraction interaction_;
  std::vector<const Var*> reduce_args_;
};

class ReducePrototype {
 public:
  ReducePrototype(
      ExprHandle init,
      ReduceInteraction& interaction,
      ReduceBodyFunc& body_func)
      : init_(init), interaction_(interaction), body_func_(body_func) {}

  ReducePrototype(ExprHandle init, ReduceInteraction& interaction, Buffer& buf)
      : init_(init),
        interaction_(interaction),
        body_func_([&buf](const std::vector<VarHandle>& v) -> ExprHandle {
          return buf.call(v);
        }) {}

  template <typename RI>
  ReducePrototype(ExprHandle init, RI interaction, Buffer& buf) : init_(init) {
    interaction_ = interaction;
    body_func_ = [&buf](ParameterList& v) -> ExprHandle { return buf.call(v); };
  }

  template <typename RI, typename BF>
  ReducePrototype(ExprHandle init, RI interaction, BF body_func) : init_(init) {
    body_func_ = body_func;
    interaction_ = interaction;
  }

  Reducer* operator()(
      Buf* result_buf,
      std::vector<const Var*> outer,
      std::vector<const Var*> inner) const {
    std::vector<const Var*> all_vars;
    all_vars.insert(all_vars.end(), outer.begin(), outer.end());
    all_vars.insert(all_vars.end(), inner.begin(), inner.end());
    ExprHandle body = body_func_(VarVectorToVarHandleVector(all_vars));

    std::vector<const Expr*> indices;
    for (size_t i = 0; i < outer.size(); i++) {
      indices.push_back(outer[i]);
    }

    ExprHandle accum =
        ExprHandle(new Load(body.dtype(), result_buf, indices, new IntImm(1)));
    Stmt* init = new Store(
        result_buf,
        indices,
        new Cast(body.dtype(), init_.node()),
        new IntImm(1));

    return new Reducer(accum, init, body, interaction_, inner);
  }

 private:
  ExprHandle init_;
  ReduceInteraction interaction_;
  ReduceBodyFunc body_func_;
};

class Sum : public ReducePrototype {
 public:
  template <typename BF>
  Sum(BF body_func)
      : ReducePrototype(
            ExprHandle(0),
            [](ExprHandle a, ExprHandle b) { return a + b; },
            body_func) {}
  Sum(Buffer& buf)
      : ReducePrototype(
            ExprHandle(0),
            [](ExprHandle a, ExprHandle b) { return a + b; },
            buf) {}
};

namespace {
ExprHandle maximumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::max());
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return ExprHandle();
}

static ExprHandle minimumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::min());
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}
} // namespace

class Maximum : public ReducePrototype {
 public:
  Maximum(ReduceBodyFunc& body_func, Dtype dtype)
      : ReducePrototype(
            minimumVal(dtype.scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Max::make(a, b, true); },
            body_func) {}
  Maximum(Buffer& buf)
      : ReducePrototype(
            minimumVal(buf.dtype().scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Max::make(a, b, true); },
            buf) {}
  template <typename BF>
  Maximum(BF body_func, ExprHandle initializer)
      : ReducePrototype(
            initializer,
            [](ExprHandle a, ExprHandle b) { return Max::make(a, b, true); },
            body_func) {}
};

class Minimum : public ReducePrototype {
 public:
  Minimum(ReduceBodyFunc& body_func, Dtype dtype)
      : ReducePrototype(
            maximumVal(dtype.scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Min::make(a, b, true); },
            body_func) {}
  Minimum(Buffer& buf)
      : ReducePrototype(
            maximumVal(buf.dtype().scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Min::make(a, b, true); },
            buf) {}
  template <typename BF>
  Minimum(BF body_func, ExprHandle initializer)
      : ReducePrototype(
            initializer,
            [](ExprHandle a, ExprHandle b) { return Min::make(a, b, true); },
            body_func) {}
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
