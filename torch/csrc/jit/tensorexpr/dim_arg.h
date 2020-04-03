#pragma once
#include <torch/csrc/jit/tensorexpr/expr.h>

namespace torch {
namespace jit {
namespace tensorexpr {
// A helper structure to store the arguments to specify dimensions. In the
// Compute arugments for dim_args, all of the following is supported. For
// example:
//    dim_args: {1, 2, 3, 4}
//    dim_args: {{1, "x"}, {2, "y"}, {3, "z"}}
//    dim_args: {1, 2, {3, "x"}}
class DimArg {
 public:
  // Intentionally leave out explicit to allow implicit conversions.
  DimArg(const ExprHandle& dim) : dim_(dim) {}
  DimArg(const ExprHandle& dim, const std::string& name_hint)
      : dim_(dim), name_hint_(name_hint) {}
  const ExprHandle& dim() const {
    return dim_;
  }
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  ExprHandle dim_;
  std::string name_hint_;
};

TORCH_API static void unpack_dim_args(
    const std::vector<DimArg>& dim_args,
    std::vector<const Expr*>* dims,
    std::vector<const Var*>* vars) {
  dims->clear();
  vars->clear();
  for (const DimArg& dim_arg : dim_args) {
    dims->push_back(dim_arg.dim().node());
    vars->push_back(new Var(dim_arg.name_hint(), kInt));
  }
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
