#pragma once

#include "dtypes.cuh"
#include "matrix.cuh"
#include <cuda_runtime.h>


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

template <size_t N>
static __global__ void _gemm_nn_block_8x8(
    fp32 *A, // row-wise
    fp32 *B, // row-wise
    fp32 *C // row-wise
    // size_t N // Number of blocks
) {
    assert(blockDim.x == 8 && blockDim.y == 8);

    size_t block_col = blockIdx.x * blockDim.x;
    size_t block_row = blockIdx.y * blockDim.y;

    // size_t global_col = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t global_row = blockIdx.y * blockDim.y + threadIdx.y;

    size_t local_x = threadIdx.x;
    size_t local_y = threadIdx.y;

    __shared__ fp32 block_b[8][8]; // row-wise
    __shared__ fp32 block_c[8][8]; // row-wise

    for (size_t k = 0; k < N; k++) {
        // block_b[local_x][local_y] = B[block_row + local_x][local_y + N*k];
        block_b[local_y][local_x] = B[block_col + local_x + (k + local_y) *N*8];
        __syncthreads();

        block_c[local_y][local_x] = 0;

        for (size_t i = 0; i < 8; i++) {
            block_c[local_y][local_x] += A[N * block_row + N * 8 * local_y + k*8 + i] * block_b[i][local_x];
        }

        C[N * block_row + 8 * N * local_y + block_col + local_x] = block_c[local_y][local_x];
        __syncthreads();
    }
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
        C[row * M + col] = 0;
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

    _gemm_nkm_simple<N, K, M><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item());
}


template <size_t N>
__host__ void _gemm_nn_block_8x8_launcher(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    assert(A.shape().first == N && A.shape().second == N);
    assert(B.shape().first == N && B.shape().second == N);
    assert(C.shape().first == N && C.shape().second == N);

    assert((N % 8) == 0);

    assert(A.get_layout() == ROW_WISE);
    assert(B.get_layout() == ROW_WISE);
    assert(C.get_layout() == ROW_WISE);

    A.cuda();
    B.cuda();
    C.cuda();

    dim3 block_dim(8, 8); // x, y
    dim3 grid_dim(N >> 3, N >> 3);

    _gemm_nn_block_8x8<N><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item());
}
