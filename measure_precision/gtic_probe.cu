// gtick_probe.cu
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long read_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void probe(unsigned long long* out, int iters) {
  // single thread to keep reads consecutive & on one SM
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  unsigned long long prev = read_globaltimer();
  for (int i = 0; i < iters; ++i) {
    unsigned long long now = read_globaltimer();
    out[i] = now - prev;    // Δt in "timer units" (ns on your stack)
    prev = now;
  }
}

int main() {
  const int iters = 200000;
  unsigned long long *d = nullptr;
  cudaMalloc(&d, iters * sizeof(unsigned long long));

  probe<<<1, 1>>>(d, iters);
  cudaDeviceSynchronize();

  std::vector<unsigned long long> h(iters);
  cudaMemcpy(h.data(), d, iters * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaFree(d);

  // drop zeros
  std::vector<unsigned long long> nz;
  nz.reserve(h.size());
  for (auto v : h) if (v != 0ULL) nz.push_back(v);

  if (nz.empty()) {
    std::puts("No non-zero samples; increase iters.");
    return 0;
  }

  std::sort(nz.begin(), nz.end());
  unsigned long long min_nz = nz.front();

  // mode via single pass on sorted data
  unsigned long long mode_val = nz[0];
  int mode_count = 1, cur_count = 1;
  for (size_t i = 1; i < nz.size(); ++i) {
    if (nz[i] == nz[i-1]) {
      ++cur_count;
    } else {
      if (cur_count > mode_count) { mode_count = cur_count; mode_val = nz[i-1]; }
      cur_count = 1;
    }
  }
  if (cur_count > mode_count) { mode_count = cur_count; mode_val = nz.back(); }

  std::printf("Samples (non-zero): %zu\n", nz.size());
  std::printf("min Δt: %llu ns\n", (unsigned long long)min_nz);
  std::printf("mode Δt: %llu ns (count=%d)\n", (unsigned long long)mode_val, mode_count);

  // show first few distinct bins
  std::puts("first few bins:");
  int shown = 0;
  for (size_t i = 0; i < nz.size() && shown < 10; ) {
    size_t j = i + 1;
    while (j < nz.size() && nz[j] == nz[i]) ++j;
    std::printf("  %6llu ns : %zu\n", (unsigned long long)nz[i], (j - i));
    ++shown;
    i = j;
  }
  return 0;
}
