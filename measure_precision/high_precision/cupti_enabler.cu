// cupti_enabler.cu
#include <cuda.h>
#include <cupti.h>
#include <stdio.h>
#define CUPTI(call) do{ CUptiResult s=(call); if(s!=CUPTI_SUCCESS){ \
  const char* e; cuptiGetResultString(s,&e); fprintf(stderr,"CUPTI: %s\n", e);} }while(0)

__attribute__((constructor))
static void enable_cupti_highres() {
  cuInit(0);
  CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  // Optional: flush immediately to keep overhead near-zero
  cuptiActivityFlushAll(0);
}
