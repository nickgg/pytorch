#include "c2_reduce.h"
#include "nnc_reduce.h"
#include "simple_reduce.h"

#include <iomanip>
#include <iostream>
#include <sstream>

#define SHOW_ERRORS 0

#define CHECK_CUDA(cmd)            \
  {                                \
    cudaError_t err = (cmd);       \
    if (err != cudaSuccess) {      \
      printf("%s failed\n", #cmd); \
      exit(-1);                    \
    }                              \
  }

template <typename Func>
std::pair<float, float> run_2d_benchmark(
    int M,
    int N,
    float* input,
    float* output,
    int outputSize,
    float* expected,
    const Func& func) {
  float *dinput, *doutput;
  CHECK_CUDA(cudaMalloc(&dinput, M * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&doutput, outputSize * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dinput, input, M * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  float milliseconds = 0;
  double total = 0;
  double maxerr = 0;
  int RUNS = 1000;

  // +1 to warm up
  for (int i = 0; i < RUNS + 1; ++i) {
    // Reset outputs
    CHECK_CUDA(cudaMemcpy(
        doutput, output, outputSize * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    func(M, N, dinput, doutput);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    // skip the first to warm up.
    if (i > 0) {
      total += milliseconds;
    }
    CHECK_CUDA(cudaMemcpy(
        output, doutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int m = 0; m < outputSize; ++m) {
      double err = fabs(output[m] - expected[m]);
      maxerr = std::max(maxerr, err);
    }
  }

  CHECK_CUDA(cudaFree(dinput));
  CHECK_CUDA(cudaFree(doutput));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return std::make_pair(1000 * (total / (float)RUNS), maxerr);
}

void print_header(std::string name) {
  std::cout << std::setw(20) << name;
  std::cout << std::setw(16) << "Caffe2";
  std::cout << std::setw(16) << "NNC";
  std::cout << std::setw(16) << "Simple";
  std::cout << std::setw(16) << "Better";
  std::cout << "\n";
}

void column_reduce_benchmark(const int M, const int N) {
  float input[M * N];
  float output[M];
  float expected[M];
  for (int m = 0; m < M; ++m) {
    output[m] = 0.0f;
    float x = 0;
    for (int n = 0; n < N; ++n) {
      input[m * N + n] = n + (float)m / (float)M;
      x += input[m * N + n];
    }
    expected[m] = x;
  }

  auto c2Context = get_c2_context();
  auto nncContext = get_2dcol_nnc_context(M, N);

  std::stringstream test_name;
  test_name << "(" << M << ", " << N << ")";
  std::cout << std::setw(20) << test_name.str();
  auto c2_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      M,
      expected,
      [&c2Context](int M, int N, float* din, float* dout) {
        caffe2_2dcol_reduce(c2Context, M, N, din, dout);
      });
  std::cout << std::setw(16) << c2_res.first;

  auto nnc_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      M,
      expected,
      [&nncContext](int M, int N, float* din, float* dout) {
        nnc_reduce(nncContext, din, dout);
      });
  std::cout << std::setw(16) << nnc_res.first;

  auto simple_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      M,
      expected,
      [](int M, int N, float* din, float* dout) {
        simpleColumnSum(M, N, din, dout);
      });
  std::cout << std::setw(16) << simple_res.first;

  auto better_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      M,
      expected,
      [](int M, int N, float* din, float* dout) {
        betterColumnSum(M, N, din, dout);
      });
  std::cout << std::setw(16) << better_res.first;
  std::cout << "\n";

#if SHOW_ERRORS
  std::cout << std::setw(20) << "error:";
  std::cout << std::setw(16) << c2_res.second;
  std::cout << std::setw(16) << nnc_res.second;
  std::cout << std::setw(16) << simple_res.second;
  std::cout << std::setw(16) << better_res.second;
  std::cout << "\n\n";
#endif
}

void row_reduce_benchmark(const int M, const int N) {
  float input[M * N];
  float output[N];
  float expected[N];
  for (int n = 0; n < N; ++n) {
    output[n] = 0.0f;
    float x = 0;
    for (int m = 0; m < M; ++m) {
      input[m * N + n] = m + (float)n / (float)N;
      x += input[m * N + n];
    }
    expected[n] = x;
  }

  auto c2Context = get_c2_context();
  auto nncContext = get_2drow_nnc_context(M, N);
  std::stringstream test_name;
  test_name << "(" << M << ", " << N << ")";
  std::cout << std::setw(20) << test_name.str();

  auto c2_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      N,
      expected,
      [&c2Context](int M, int N, float* din, float* dout) {
        caffe2_2drow_reduce(c2Context, M, N, din, dout);
      });
  std::cout << std::setw(16) << c2_res.first;

  auto nnc_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      N,
      expected,
      [&nncContext](int M, int N, float* din, float* dout) {
        nnc_reduce(nncContext, din, dout);
      });
  std::cout << std::setw(16) << nnc_res.first;

  auto simple_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      N,
      expected,
      [](int M, int N, float* din, float* dout) {
        simpleRowSum(M, N, din, dout);
      });
  std::cout << std::setw(16) << simple_res.first;

  auto better_res = run_2d_benchmark(
      M,
      N,
      input,
      output,
      N,
      expected,
      [](int M, int N, float* din, float* dout) {
        betterRowSum(M, N, din, dout);
      });
  std::cout << std::setw(16) << better_res.first;
  std::cout << "\n";

#if SHOW_ERRORS
  std::cout << std::setw(20) << "error:";
  std::cout << std::setw(16) << c2_res.second;
  std::cout << std::setw(16) << nnc_res.second;
  std::cout << std::setw(16) << simple_res.second;
  std::cout << std::setw(16) << better_res.second;
  std::cout << "\n\n";
#endif
}

int main(void) {
  KernelScope kernel_scope;

  print_header("Column sum");
  std::cout << std::setprecision(5);
  column_reduce_benchmark(10, 100);
  column_reduce_benchmark(100, 100);
  column_reduce_benchmark(100, 10000);
  column_reduce_benchmark(1000, 1000);
  std::cout << "\n";
  print_header("Row sum");
  row_reduce_benchmark(10, 100);
  row_reduce_benchmark(100, 100);
  row_reduce_benchmark(100, 10000);
  row_reduce_benchmark(1000, 1000);

  return 0;
}

