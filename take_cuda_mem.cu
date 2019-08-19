#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    auto __cond__ = (cond);                                                    \
    if (__cond__ != cudaSuccess) {                                             \
      std::string __msg__ = "Runtime error in " #cond;                         \
      __msg__ +=                                                               \
          ", error code is " + std::to_string(static_cast<int>(__cond__));     \
      throw std::runtime_error(__msg__);                                       \
    }                                                                          \
  } while (false)

static __global__ void FillConstantKernel(uint8_t *p, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    p[idx] = static_cast<uint8_t>(idx) & 0xFF;
  }
}

static void CudaFree(void *p) { CUDA_CHECK(cudaFree(p)); }

static void ThreadMain(int dev_id, uint8_t *p, int n, cudaStream_t stream) {
  CUDA_CHECK(cudaSetDevice(dev_id));
  size_t idx = 0;
  while (1) {
    int thread_num = 512;
    int grid_num = (n + thread_num - 1) / thread_num;
    int r = rand() % 10 + 1;
    int m = n * r;
    FillConstantKernel<<<grid_num, thread_num, 0, stream>>>(p, m);
    if (idx % 1 == 0) {
      idx = 0;
      CUDA_CHECK(cudaStreamSynchronize(stream));

      std::this_thread::sleep_for(std::chrono::microseconds(100 * r));
    }
    ++idx;
  }
}

int main() {
  int dev_cnt = -1;
  CUDA_CHECK(cudaGetDeviceCount(&dev_cnt));

  if (dev_cnt <= 0) {
    std::cerr << "Error! Please set CUDA_VISIBLE_DEVICES before running!"
              << std::endl;
    return -1;
  }

  std::cout << "Device number: " << dev_cnt << std::endl;

  // Malloc 4 GiB GPU and launch kernel
  size_t bytes = (static_cast<size_t>(4) << 30);

  std::vector<std::unique_ptr<uint8_t, void (*)(void *)>> ptrs;
  std::vector<cudaStream_t> streams;

  ptrs.reserve(dev_cnt);
  streams.resize(dev_cnt);

  for (int i = 0; i < dev_cnt; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    uint8_t *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    ptrs.emplace_back(p, CudaFree);
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  // Launch several kernels and each device
  std::vector<std::thread> threads;
  threads.reserve(dev_cnt);
  for (int i = 0; i < dev_cnt; ++i) {
    auto *p = ptrs[i].get();
    threads.emplace_back(ThreadMain, i, p, static_cast<int>(bytes >> 5),
                         streams[i]);
  }

  for (auto &th : threads) {
    th.join();
  }

  for (auto &stream : streams) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  return 0;
}
