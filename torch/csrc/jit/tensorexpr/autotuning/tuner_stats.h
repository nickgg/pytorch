#pragma once

#include <atomic>
#include <chrono>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {

struct TuningStats {
  size_t candidates_generated{0};
  size_t candidates_resolved{0};
  size_t total_runs{0};

  std::chrono::milliseconds running_time{0};
  std::chrono::milliseconds codegen_time{0};
  std::chrono::milliseconds generation_time{0};
  std::chrono::milliseconds sorting_time{0};
  std::chrono::milliseconds checking_time{0};
};

template <typename It>
std::chrono::milliseconds TO_MS(It t) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(t);
}

template <typename It>
std::chrono::microseconds TO_US(It t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(t);
}

static std::chrono::time_point<std::chrono::high_resolution_clock>
host_timestamp() {
  // Fences around the timestamp to prevent reordering before/after the
  // timestamp.
  std::atomic_signal_fence(std::memory_order_seq_cst);
  auto now = std::chrono::high_resolution_clock::now();
  std::atomic_signal_fence(std::memory_order_seq_cst);
  return now;
}

} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch

