#pragma once

#include <cuda_runtime.h>

#include "dtypes.cuh"
#include "matrix.cuh"
#include "utils.cuh"


/*

RTX 3060 math https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-12-gb.c3682
 
SM-count 28

Warp-size 32 threads
Active warps in SM - 4
Total warps active - 112

Max waprs in SM - 48
Max Block per SM - 16!!

Tensor Core count - 112

SMEM per SM - 128kb
equal SMEM per Warp - 32kb

*/

template <size_t N, size_t BS>
static __global__ void _gemm_nnn_block_simple(
    fp32 *A, // row-wise
    fp32 *B, // row-wise
    fp32 *C // row-wise
) {
    assert(blockDim.x == BS && blockDim.y == BS);

    size_t blockPtr = N * (blockIdx.y * blockDim.y) + blockDim.x * blockIdx.x;
    size_t threadPtr = blockPtr + N * threadIdx.y + threadIdx.x;

    fp32 sum = 0.0;

    __shared__ fp32 block_a[BS][BS];
    __shared__ fp32 block_b[BS][BS];

    for (size_t s = 0; s < (N / BS); s++) {
        size_t a_block_ptr = N * (blockIdx.y * blockDim.y) + s * blockDim.x;
        size_t b_block_ptr = N * blockDim.y * s + blockIdx.x * blockDim.x;

        block_a[threadIdx.y][threadIdx.x] = A[a_block_ptr + threadIdx.x + threadIdx.y * N];
        block_b[threadIdx.y][threadIdx.x] = B[b_block_ptr + threadIdx.x + threadIdx.y * N];

        __syncthreads();

        for (size_t k = 0; k < BS; k++) {
            // size_t loc_a_ptr = a_block_ptr + N * threadIdx.y + k;
            // size_t loc_b_ptr = b_block_ptr + threadIdx.x + N * k;

            // sum += A[loc_a_ptr] * B[loc_b_ptr];

            sum += block_a[threadIdx.y][k] * block_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[threadPtr] = sum;
}


template <size_t N, size_t K, size_t M> // n*k x k*m = n*m
static __global__ void _gemm_nkm_simple(
    fp32 *A, // row-wise
    fp32 *B, // row-wise
    fp32 *C // row-wise
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        fp32 sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k*M + col];
        }
        C[row * M + col] = sum;
    }
}

template <size_t N, size_t K, size_t M> // n*k x k*m = n*m
__host__ void _gemm_nkm_simple_launcher(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    assert(A.shape().first == N && A.shape().second == K);
    assert(B.shape().first == K && B.shape().second == M);
    assert(C.shape().first == N && C.shape().second == M);

    assert(A.get_layout() == ROW_WISE);
    assert(B.get_layout() == ROW_WISE);
    assert(C.get_layout() == ROW_WISE);

    A.cuda();
    B.cuda();
    C.cuda();

    dim3 block_dim(16, 16);
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x,
                  (N + block_dim.y - 1) / block_dim.y);

    cudaFuncSetCacheConfig(_gemm_nkm_simple<N, K, M>, cudaFuncCachePreferL1);
    _gemm_nkm_simple<N, K, M><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item());
    CUDA_CHECK(cudaDeviceSynchronize());
}


template <size_t N, size_t BS>
__host__ void _gemm_nn_block_launcher(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    assert(A.shape().first == N && A.shape().second == N);
    assert(B.shape().first == N && B.shape().second == N);
    assert(C.shape().first == N && C.shape().second == N);

    assert((N % BS) == 0);

    assert(A.get_layout() == ROW_WISE);
    assert(B.get_layout() == ROW_WISE);
    assert(C.get_layout() == ROW_WISE);

    A.cuda();
    B.cuda();
    C.cuda();

    dim3 block_dim(BS, BS); // x, y
    dim3 grid_dim(N / BS, N / BS);

    cudaFuncSetCacheConfig(_gemm_nnn_block_simple<N, BS>, cudaFuncCachePreferShared);
    _gemm_nnn_block_simple<N, BS><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item());
    CUDA_CHECK(cudaDeviceSynchronize());
}
