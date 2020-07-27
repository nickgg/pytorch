#pragma once

#include <caffe2/core/context_gpu.h>
#include <caffe2/utils/math/reduce.h>

struct C2Context {
  caffe2::CUDAContext context;
};

C2Context get_c2_context() {
  return C2Context();
}

void caffe2_2dcol_reduce(
    C2Context& context,
    const int M,
    const int N,
    float* input,
    float* output) {
  int in_dims[2] = {M, N};
  int out_dims[2] = {M, 1};
  caffe2::math::ReduceSum<float, caffe2::CUDAContext>(
      2, in_dims, out_dims, 1, input, output, &context.context);
}

void caffe2_2drow_reduce(
    C2Context& context,
    const int M,
    const int N,
    float* input,
    float* output) {
  int in_dims[2] = {M, N};
  int out_dims[2] = {1, N};
  caffe2::math::ReduceSum<float, caffe2::CUDAContext>(
      2, in_dims, out_dims, 1, input, output, &context.context);
}

