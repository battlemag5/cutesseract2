#pragma once

#include <cuda_runtime.h>

#include "dtypes.cuh"
#include "tensor.cuh"
#include "utils.cuh"

// ==========================================================================
//  Strided GEMM: contraction over last axis of A and first axis of B
//  A shape: [..., N, K]  B shape: [K, M, ...]  C shape: [..., N, M, ...]
// ==========================================================================

template <typename T>
__global__ void gemm_nkm_strided_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    size_t N, size_t K, size_t M,
    size_t stride_AN, size_t stride_AK,
    size_t stride_BK, size_t stride_BM,
    size_t stride_CN, size_t stride_CM
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * stride_AN + k * stride_AK] *
                   B[k * stride_BK + col * stride_BM];
        }
        C[row * stride_CN + col * stride_CM] = sum;
    }
}

template <typename T>
__host__ void gemm_nkm_strided(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C
) {
    size_t ndim_A = A.get_ndim();
    size_t ndim_B = B.get_ndim();

    size_t N = A.get_shape(ndim_A - 2);
    size_t K = A.get_shape(ndim_A - 1);
    size_t M = B.get_shape(1);

    assert(B.get_shape(0) == K);

    size_t stride_AN = A.get_stride(ndim_A - 2);
    size_t stride_AK = A.get_stride(ndim_A - 1);
    size_t stride_BK = B.get_stride(0);
    size_t stride_BM = B.get_stride(1);
    size_t stride_CN = C.get_stride(0);
    size_t stride_CM = C.get_stride(1);

    dim3 block_dim(16, 16);
    dim3 grid_dim((M + 15) / 16, (N + 15) / 16);

    gemm_nkm_strided_kernel<T><<<grid_dim, block_dim>>>(
        A.item(), B.item(), C.item(),
        N, K, M,
        stride_AN, stride_AK,
        stride_BK, stride_BM,
        stride_CN, stride_CM
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
