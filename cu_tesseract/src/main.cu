#include "matrix.cuh"
#include "kernels.cuh"

#include <iostream>
#include <chrono>

using std::cout;
using std::endl;

constexpr size_t n = 3072, k = 3072, m = 3072;
// constexpr size_t N = 256;

void warmup() {
    auto A = Matrix<fp32>((size_t)256, (size_t)256, ROW_WISE, CUDA);
    auto B = Matrix<fp32>((size_t)256, (size_t)256, ROW_WISE, CUDA);
    auto C = Matrix<fp32>((size_t)256, (size_t)256, ROW_WISE, CUDA);

    A.fill_random();
    B.fill_random();

    _gemm_nkm_simple_launcher<256, 256, 256>(A, B, C);
}


void verify_cpu(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    A.cpu();
    B.cpu();
    C.cpu();

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            fp32 sum = 0.0;
            for (size_t r = 0; r < k; r++) {
                sum += A.get(i, r) * B.get(r, j);
            }

            if (std::abs(sum - C.get(i, j)) >= 1e-3) {
                cout << sum << ' ' << C.get(i, j) << " (" << i << ", " << j << ")\n";
                throw std::runtime_error("verification failed");
            }
        }
    }
}

std::chrono::duration<double, std::milli> test_blockwise() {
    auto A = Matrix<fp32>((size_t)n, (size_t)k, ROW_WISE, CUDA);
    auto B = Matrix<fp32>((size_t)k, (size_t)m, ROW_WISE, CUDA);
    auto C = Matrix<fp32>((size_t)n, (size_t)m, ROW_WISE, CUDA);

    A.fill_random(911ull);
    B.fill_random(911ull);

    // A.ones();
    // B.ones();

    // A.cuda();
    // B.cuda();

    // warmup();
    auto start_time = std::chrono::high_resolution_clock::now();

    _gemm_nn_block_8x8_launcher<n>(A, B, C);

    return std::chrono::high_resolution_clock::now() - start_time;

    // verify_cpu(A, B, C);
}

std::chrono::duration<double, std::milli> test_elementwise() {
    auto A = Matrix<fp32>((size_t)n, (size_t)k, ROW_WISE, CUDA);
    auto B = Matrix<fp32>((size_t)k, (size_t)m, ROW_WISE, CUDA);
    auto C = Matrix<fp32>((size_t)n, (size_t)m, ROW_WISE, CUDA);

    A.fill_random(911ull);
    B.fill_random(911ull);

    // warmup();
    auto start_time = std::chrono::high_resolution_clock::now();

    _gemm_nkm_simple_launcher<n, k, m>(A, B, C);

    return std::chrono::high_resolution_clock::now() - start_time;

    // verify_cpu(A, B, C);
}

signed main() {

    size_t num_tries = 16;

    std::chrono::duration<double, std::milli> min_block = test_blockwise();
    std::chrono::duration<double, std::milli> min_element = test_elementwise();

    for (size_t i = 0; i < num_tries; i++) {
        std::chrono::duration<double, std::milli> cur = test_blockwise();
        if (cur < min_block) {
            min_block = cur;
        }
    }

    cout << "Blockwise GPU multiplication duration: ~" << min_block << "\n";

    for (size_t i = 0; i < num_tries; i++) {
        std::chrono::duration<double, std::milli> cur = test_elementwise();
        if (cur < min_element) {
            min_element = cur;
        }
    }

    cout << "Elementwise GPU multiplication duration: ~" << min_element << "\n";

    return 0;
}
