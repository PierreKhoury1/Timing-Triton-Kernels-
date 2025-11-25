#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cupti.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    }

#define CHECK_CUPTI(err) \
    if (err != CUPTI_SUCCESS) { \
        const char* errstr; cuptiGetResultString(err, &errstr); \
        printf("CUPTI error: %s\n", errstr); exit(1); \
    }

/* --------------------------------------------------------------------------
   SIMPLE CUDA KERNEL
   -------------------------------------------------------------------------- */
__global__ void add_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];   // simple FADD
}

/* --------------------------------------------------------------------------
   RUN KERNELS — this will be called from main()
   -------------------------------------------------------------------------- */
extern "C" void run_kernels() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Launch kernel multiple times to generate PC samples
    for (int i = 0; i < 50; i++) {
        add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    float *h_c = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("Result sample: %f\n", h_c[0]);

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

/* --------------------------------------------------------------------------
   CUPTI PC SAMPLING CALLBACKS
   -------------------------------------------------------------------------- */
void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 16 * 1024;
    *buffer = (uint8_t*) malloc(*size);
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
    uint8_t* buffer, size_t size, size_t validSize) {

    CUpti_Activity* record = NULL;

    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_PC_SAMPLING) {
            CUpti_ActivityPCSampling* pcs =
                (CUpti_ActivityPCSampling*) record;

            printf("PC: 0x%llx  samples=%u  stall=%u  state=%u\n",
                (unsigned long long) pcs->pcOffset,
                pcs->samplingData.samples,
                pcs->samplingData.stallReason,
                pcs->samplingData.state);
        }
    }

    free(buffer);
}

/* --------------------------------------------------------------------------
   MAIN
   -------------------------------------------------------------------------- */
int main() {
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
    CHECK_CUPTI(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Enable maximum sampling frequency
    CHECK_CUPTI(cuptiActivityConfigurePCSampling(
        RANDOM_MODE,
        CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN
    ));

    printf("Running kernels...\n");
    run_kernels();

    printf("Flushing CUPTI...\n");
    CHECK_CUPTI(cuptiActivityFlushAll(0));
    return 0;
}

