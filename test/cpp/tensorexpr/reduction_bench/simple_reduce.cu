#include <iostream>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>


__global__ void simpleColumnSum_impl(
    const int M,
    const int N,
    const float* input,
    float* output) {
  float x = 0;
  const float* p = &input[blockIdx.x * N];

  for (int i = 0; i < N; i += 1) {
    x += p[i];
  }
  output[blockIdx.x] = x;
}

__global__ void betterColumnSum_impl(
    const int M,
    const int N,
    const float* input,
    float* output) {
  extern __shared__ float shared[];

  float x = 0;
  const float* p = &input[blockIdx.x * N];

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    x += p[i];
  }

  shared[threadIdx.x] = x;
  __syncthreads();

  // shuffle down
  for(int offset = blockDim.x / 2; offset > 0; offset = offset >> 1) {
      if(threadIdx.x < offset) {
          shared[threadIdx.x] += shared[threadIdx.x + offset];
      }
      __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = shared[0];
  }
}

__global__ void simpleRowSum_impl(
    const int M,
    const int N,
    const float* input,
    float* output) {
  float x = 0;
  const float* p = &input[blockIdx.x];

  for (int i = 0; i < M*N; i += N) {
    x += p[i];
  }
  output[blockIdx.x] = x;
}

__global__ void betterRowSum_impl(const int M, const int N, float *batch,
                                    float *dest) {
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const int n = blockIdx.x;
  float val = 0;

  for (int m = threadIdx.x; m < M; m += blockDim.x) {
    int index = m * N + n;
    val += batch[index];
  }

  val = BlockReduce(temp_storage).Reduce(val, cub::Sum());
  // The call to Reduce synchronizes all threads, so val contains the final
  // result.
  if (threadIdx.x == 0) {
    dest[n] = val;
  }
}


void betterColumnSum(const int M, const int N, float* input, float* output) {
  dim3 blocks(M);
  dim3 threads(256);
  size_t mem = sizeof(float) * (size_t)threads.x;
  betterColumnSum_impl<<<blocks, threads, mem>>>(M, N, input, output);
}

void simpleColumnSum(const int M, const int N, float* input, float* output) {
  dim3 blocks(M);
  dim3 threads(1);
  simpleColumnSum_impl<<<blocks, threads>>>(M, N, input, output);
}

void betterRowSum(const int M, const int N, float* input, float* output) {
  dim3 blocks(N);
  dim3 threads(256);
  betterRowSum_impl<<<blocks, threads>>>(M, N, input, output);
}

void simpleRowSum(const int M, const int N, float* input, float* output) {
  dim3 blocks(N);
  dim3 threads(1);
  simpleRowSum_impl<<<blocks, threads>>>(M, N, input, output);
}
