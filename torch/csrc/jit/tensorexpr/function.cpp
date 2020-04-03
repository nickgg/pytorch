#include <torch/csrc/jit/tensorexpr/function.h>

#include <c10/util/Logging.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/dim_arg.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func) {
  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body = body_func(VarVectorToVarHandleVector(args)).node();
  Function* func = new Function(func_name, dims, args, body);
  const Buf* buf = func->func_var(0);
  return new Tensor(buf, func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&)>& body_func) {
  if (dim_args.size() != 1) {
    throw malformed_input();
  }

  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body = body_func(VarHandle(args[0])).node();
  Function* func = new Function(func_name, dims, args, body);
  const Buf* buf = func->func_var(0);
  return new Tensor(buf, func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func) {
  if (dim_args.size() != 2) {
    throw malformed_input();
  }
  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body = body_func(VarHandle(args[0]), VarHandle(args[1])).node();
  Function* func = new Function(func_name, dims, args, body);
  const Buf* buf = func->func_var(0);
  return new Tensor(buf, func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func) {
  if (dim_args.size() != 3) {
    throw malformed_input();
  }
  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body =
      body_func(VarHandle(args[0]), VarHandle(args[1]), VarHandle(args[2]))
          .node();
  Function* func = new Function(func_name, dims, args, body);
  const Buf* buf = func->func_var(0);
  return new Tensor(buf, func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func) {
  if (dim_args.size() != 4) {
    throw malformed_input();
  }
  std::vector<const Expr*> dims;
  std::vector<const Var*> args_nodes;
  unpack_dim_args(dim_args, &dims, &args_nodes);
  auto args = VarVectorToVarHandleVector(args_nodes);
  const Expr* body = body_func(args[0], args[1], args[2], args[3]).node();
  Function* func = new Function(func_name, dims, args_nodes, body);
  const Buf* buf = func->func_var(0);
  return new Tensor(buf, func, 0);
}

TORCH_API Tensor* Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const ReducePrototype& reduce_proto,
    const std::vector<DimArg>& reduce_args) {
  std::vector<const Expr*> dims;
  std::vector<const Var*> vars;
  unpack_dim_args(dim_args, &dims, &vars);

  std::vector<const Expr*> reduce_dims;
  std::vector<const Var*> reduce_vars;
  unpack_dim_args(reduce_args, &reduce_dims, &reduce_vars);

  std::vector<const Var*> all_vars;
  all_vars.insert(all_vars.end(), vars.begin(), vars.end());
  all_vars.insert(all_vars.end(), reduce_vars.begin(), reduce_vars.end());

  Buf* func_result = new Buf(new Var(func_name, kHandle), dims);

  const Reducer* reducer = reduce_proto(func_result, vars, reduce_vars);
  dims.insert(dims.end(), reduce_dims.begin(), reduce_dims.end());
  Function* func =
      new Function(func_name, func_result, dims, all_vars, reducer);
  return new Tensor(func_result, func, 0);
}

Stmt* Function::ElementStmt(size_t index) {
  const Buf* buf = func_var(index);
  std::vector<const Expr*> indices;
  for (size_t i = 0; i < buf->ndim(); i++) {
    indices.push_back(this->args_[i]);
  }

  const Expr* mask = new IntImm(1);

  Stmt* update_stmt = new Store(buf, indices, body(index), mask);
  return update_stmt;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
