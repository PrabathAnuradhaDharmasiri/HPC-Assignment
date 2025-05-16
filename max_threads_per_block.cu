#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    const int N = 1024 * 10; // Total number of elements
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    size_t size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Hardcoded first 10 values
    float tempA[10] = {145, 560, 832, 307, 412, 150, 620, 978, 294, 80};
    float tempB[10] = {876, 739, 128, 683, 413, 215, 175, 101, 606, 519};

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        if (i < 10) {
            a[i] = tempA[i];
            b[i] = tempB[i];
        } else {
            a[i] = rand() % 1000;
            b[i] = rand() % 1000;
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // Copy input data to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    bool errorPrinted = false;

    for (int threadsPerBlock = 1; threadsPerBlock <= 2048; threadsPerBlock++) {
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the kernel
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (!errorPrinted) {
                printf("\n🚫 Error: Kernel could not launch with %d threads per block.\n", threadsPerBlock);
                printf("✔️  Maximum supported threads per block on this GPU: %d\n", threadsPerBlock - 1);
                errorPrinted = true;
            }
            break;
        }
    }

    // Only copy and print results for one valid configuration
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    printf("\n✅ Sample output for N=%d elements:\n", N);
    for (int i = 0; i < 10; i++) {
        printf("%.1f + %.1f = %.1f\n", a[i], b[i], c[i]);
    }

    // Free memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
