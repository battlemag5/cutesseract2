
#include "dtypes.cuh"
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

__global__ _gemm_nn_block_8x8(float *A, float *B, float *C, size_t N);
