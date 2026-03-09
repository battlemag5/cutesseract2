
#include "kernels.cuh"
#include "matrix.cuh"

#include <cuda_runtime.h>

__global__ _gemm_nn_block_8x8(
    float *A, // row-wise
    float *B, // col-wise
    float *C, // row-wise
    size_t N // Number of blocks
) {
    assert(threadDim.x == 8 && threadDim.y == 8);

    size_t block_col = blockIdx.x * blockDim.x;
    size_t block_row = blockIdx.y * blockDim.y;

    // size_t global_col = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t global_row = blockIdx.y * blockDim.y + threadIdx.y;

    size_t local_x = threadIdx.x;
    size_t local_y = threadIdx.y;

    __shared__ float[8][8] block_b = {0}; // row-wise
    __shared__ float[8][8] block_c = {0}; // row-wise

    for (size_t k = 0; k < N; k++) {
        block_b[local_x][local_y] = B[block_row + local_x][local_y + N*k];
        __syncthreads();

        for (size_t i = 0; i < N; i++) {
            block_c[local_y][local_x] += A[block_row + local_y][block_col + i] * block_b[i][local_x]
        }

        C[block_row + local_y][block_col + local_x] = block_c[local_y][local_x];
        __syncthreads();
    }
}

template<size_t N>
__host__ _gemm_nn_block_8x8_launcher(Matrix &A, Matrix &B, Matrix &C) {
    assert(A.shape().first == N && A.shape().second == N);
    assert(B.shape().first == N && B.shape().second == N);
    assert(C.shape().first == N && C.shape().second == N);

    assert((N % 8) == 0);

    A.cuda();
    B.cuda();
    C.cuda();

    dim3 block_dim(8, 8); // x, y
    dim3 grid_dim(N >> 3, N >> 3);

    _gemm_nn_block_8x8<<<grid_dim, block_dim>>>(A.item(), B.item(), C.item(), N);
}
