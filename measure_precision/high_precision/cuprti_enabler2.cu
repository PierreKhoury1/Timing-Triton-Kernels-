// cupti_enabler.cu
#include <cuda.h>
#include <cupti.h>
#include <dlfcn.h>
#include <stdio.h>

#define CUPTI(call) do { CUptiResult s=(call); if(s!=CUPTI_SUCCESS){ \
  const char* e; cuptiGetResultString(s,&e); fprintf(stderr,"CUPTI: %s\n", e); } } while(0)

typedef CUresult (*cuInit_t)(unsigned int);

__attribute__((constructor))
static void enable_cupti_highres() {
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
  if (!h) { fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror()); return; }

  cuInit_t p_cuInit = (cuInit_t)dlsym(h, "cuInit");
  if (!p_cuInit) { fprintf(stderr, "dlsym cuInit failed: %s\n", dlerror()); return; }

  if (p_cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return; }

  CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  cuptiActivityFlushAll(0);
}

