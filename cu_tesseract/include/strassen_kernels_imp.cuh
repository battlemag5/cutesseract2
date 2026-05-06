#pragma once

#include <cuda_runtime.h>
#include "dtypes.cuh"

/**
 * This kernel is intended for use in Strassen's algorithm where sub-matrices
 * share the same physical buffer and leading dimension as their parents.
 */
template <typename T>
__global__ void _gemm_strassen_leaf_kernel(const T *A, size_t lda,
                                            const T *B, size_t ldb,
                                            T *C, size_t ldc,
                                            size_t N, size_t BS) {
    size_t row = blockIdx.y * BS + threadIdx.y;
    size_t col = blockIdx.x * BS + threadIdx.x;

    double sum = 0.0;

    extern __shared__ char smem[];
    T *block_a = (T *)smem;
    T *block_b = (T *)(smem + BS * BS * sizeof(T));

    for (size_t s = 0; s < (N / BS); s++) {
        // Load block from A
        if (row < N && (s * BS + threadIdx.x) < N) {
            block_a[threadIdx.y * BS + threadIdx.x] = A[row * lda + (s * BS + threadIdx.x)];
        } else {
            block_a[threadIdx.y * BS + threadIdx.x] = 0;
        }

        // Load block from B
        if ((s * BS + threadIdx.y) < N && col < N) {
            block_b[threadIdx.y * BS + threadIdx.x] = B[(s * BS + threadIdx.y) * ldb + col];
        } else {
            block_b[threadIdx.y * BS + threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial sum
        for (size_t k = 0; k < BS; k++) {
            sum += (double)block_a[threadIdx.y * BS + k] * (double)block_b[k * BS + threadIdx.x];
        }

        __syncthreads();
    }

    // Write back result
    if (row < N && col < N) {
        C[row * ldc + col] = (T)sum;
    }
}
