// cupti_inject.cu
#include <cuda.h>
#include <cupti.h>
#include <cstdio>
#include <cstdlib>

// CUPTI needs activity buffers to be provided; otherwise enabling has no effect.
static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  const size_t SZ = 1 << 20; // 1 MiB
  *buffer = (uint8_t*)std::malloc(SZ);
  *size = SZ;
  *maxNumRecords = 0; // no limit
}
static void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t) {
  std::free(buffer);
}

static void enable(CUpti_ActivityKind k, const char* name) {
  CUptiResult r = cuptiActivityEnable(k);
  const char* s = nullptr;
  if (r != CUPTI_SUCCESS) cuptiGetResultString(r, &s);
  std::fprintf(stderr, "[cupti_inject] enable %-20s -> %s\n", name, r == CUPTI_SUCCESS ? "OK" : (s ? s : "ERR"));
}

// CUDA will call this at process startup when CUDA_INJECTION64_PATH points to this .so
extern "C" void InitializeInjection() {
  std::fprintf(stderr, "[cupti_inject] InitializeInjection()\n");

  // Make sure the driver is initialized
  if (cuInit(0) != CUDA_SUCCESS) {
    std::fprintf(stderr, "[cupti_inject] cuInit failed\n");
    return;
  }

  // Register activity callbacks FIRST (critical)
  CUptiResult rr = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  if (rr != CUPTI_SUCCESS) {
    const char* s=nullptr; cuptiGetResultString(rr,&s);
    std::fprintf(stderr, "[cupti_inject] register callbacks -> %s\n", s?s:"ERR");
  }

  // Enable common kinds (some may fail depending on your CUPTI version; that’s fine)
  enable(CUPTI_ACTIVITY_KIND_RUNTIME,           "RUNTIME");
  enable(CUPTI_ACTIVITY_KIND_DRIVER,            "DRIVER");
  enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, "CONCURRENT_KERNEL");
  // One of these kernel kinds should exist on your CUPTI:
  enable(CUPTI_ACTIVITY_KIND_KERNEL,            "KERNEL");

  cuptiActivityFlushAll(0);
  std::fprintf(stderr, "[cupti_inject] CUPTI activity enabled\n");
}
