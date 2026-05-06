#pragma once

#include <cuda_runtime.h>
#include "dtypes.cuh"

/**
 * @brief Simple GEMM kernel with double precision accumulation and boundary guards.
 */
template <typename T>
__global__ void _gemm_nkm_verified_kernel(const T *A, const T *B, T *C, 
                                           size_t N, size_t K, size_t M) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        double sum = 0.0;
        for (size_t i = 0; i < K; i++) {
            sum += (double)A[row * K + i] * (double)B[i * M + col];
        }
        C[row * M + col] = (T)sum;
    }
}

/**
 * @brief Blocked GEMM kernel with double precision accumulation and boundary guards.
 */
template <typename T>
__global__ void _gemm_nn_blocked_verified_kernel(const T *A, const T *B, T *C, 
                                                  size_t N, size_t BS) {
    size_t row = blockIdx.y * BS + threadIdx.y;
    size_t col = blockIdx.x * BS + threadIdx.x;

    double sum = 0.0;

    extern __shared__ char smem[];
    T *block_a = (T *)smem;
    T *block_b = (T *)(smem + BS * BS * sizeof(T));

    for (size_t s = 0; s < (N + BS - 1) / BS; s++) {
        // Load block from A with boundary guards
        if (row < N && (s * BS + threadIdx.x) < N) {
            block_a[threadIdx.y * BS + threadIdx.x] = A[row * N + (s * BS + threadIdx.x)];
        } else {
            block_a[threadIdx.y * BS + threadIdx.x] = 0;
        }

        // Load block from B with boundary guards
        if ((s * BS + threadIdx.y) < N && col < N) {
            block_b[threadIdx.y * BS + threadIdx.x] = B[(s * BS + threadIdx.y) * N + col];
        } else {
            block_b[threadIdx.y * BS + threadIdx.x] = 0;
        }

        __syncthreads();

        for (size_t k = 0; k < BS; k++) {
            sum += (double)block_a[threadIdx.y * BS + k] * (double)block_b[k * BS + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = (T)sum;
    }
}

// Launchers for the verified kernels
template <typename T>
void _gemm_nkm_verified_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
    size_t N = A.shape().first;
    size_t K = A.shape().second;
    size_t M = B.shape().second;
    
    dim3 block_dim(16, 16);
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y);
    
    _gemm_nkm_verified_kernel<T><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item(), N, K, M);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void _gemm_nn_blocked_verified_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
    size_t N = A.shape().first;
    size_t BS = 16;
    
    dim3 block_dim(BS, BS);
    dim3 grid_dim((N + BS - 1) / BS, (N + BS - 1) / BS);
    size_t shared_mem_size = 2 * BS * BS * sizeof(T);
    
    _gemm_nn_blocked_verified_kernel<T><<<grid_dim, block_dim, shared_mem_size>>>(A.item(), B.item(), C.item(), N, BS);
    CUDA_CHECK(cudaDeviceSynchronize());
}
