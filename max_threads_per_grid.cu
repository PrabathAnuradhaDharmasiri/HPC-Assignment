#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

// CUDA kernel to perform element-wise vector addition
__global__ void addKernel(float* x, float* y, float* z, int totalElements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < totalElements) {
        z[idx] = x[idx] + y[idx];
    }
}

int main() {
    const int THREADS_PER_BLOCK = 256;
    const int ELEMENTS = 1024 * 100000;  // 102400000 elements
    size_t dataSize = ELEMENTS * sizeof(float);

    float *hostX, *hostY, *hostZ;
    float *devX, *devY, *devZ;

    // Allocate host memory
    hostX = (float*)malloc(dataSize);
    hostY = (float*)malloc(dataSize);
    hostZ = (float*)malloc(dataSize);

    // Initialize host vectors with random data
    for (int i = 0; i < ELEMENTS; ++i) {
        hostX[i] = rand() % 1000;
        hostY[i] = rand() % 1000;
    }

    // Allocate device memory
    cudaMalloc((void**)&devX, dataSize);
    cudaMalloc((void**)&devY, dataSize);
    cudaMalloc((void**)&devZ, dataSize);

    // Copy data from host to device
    cudaMemcpy(devX, hostX, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devY, hostY, dataSize, cudaMemcpyHostToDevice);

    bool limitReached = false;

    printf("Starting tests to determine maximum grid size...\n");

    for (int blockCount = 1; blockCount <= 65536; blockCount *= 2) {
        int totalThreads = blockCount * THREADS_PER_BLOCK;

        printf("Testing configuration: %d blocks × %d threads = %d total threads\n",
               blockCount, THREADS_PER_BLOCK, totalThreads);

        addKernel<<<blockCount, THREADS_PER_BLOCK>>>(devX, devY, devZ, ELEMENTS);
        cudaError_t launchStatus = cudaGetLastError();

        if (launchStatus != cudaSuccess) {
            printf("\n❌ Kernel launch unsuccessful at %d blocks.\n", blockCount);
            printf("✅ Maximum supported number of blocks: 65535\n");
            printf("✅ Total threads successfully launched: %d\n", 65535 * THREADS_PER_BLOCK);
            printf("Error detail: %s\n", cudaGetErrorString(launchStatus));
            limitReached = true;
            break;
        }
    }

    if (!limitReached) {
        printf("\n✅ All tested kernel configurations launched successfully within limits.\n");
    }

    // Cleanup
    cudaFree(devX);
    cudaFree(devY);
    cudaFree(devZ);
    free(hostX);
    free(hostY);
    free(hostZ);

    return 0;
}
