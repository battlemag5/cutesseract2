#include "matrix.cuh"
#include "kernels.cuh"
#include <iostream>

using std::cout;
using std::endl;

constexpr size_t n = 256, k = 1024, m = 256;

void warmup() {
    auto A = Matrix<fp32>((size_t)8, (size_t)8, ROW_WISE, CUDA);
    auto B = Matrix<fp32>((size_t)8, (size_t)8, ROW_WISE, CUDA);
    auto C = Matrix<fp32>((size_t)8, (size_t)8, ROW_WISE, CUDA);

    A.fill_random();
    B.fill_random();

    _gemm_nkm_simple_launcher<8, 8, 8>(A, B, C);
}

void test2() {
    warmup();

    auto A = Matrix<fp32>((size_t)n, (size_t)k, ROW_WISE, CUDA);
    auto B = Matrix<fp32>((size_t)k, (size_t)m, ROW_WISE, CUDA);
    auto C = Matrix<fp32>((size_t)n, (size_t)m, ROW_WISE, CUDA);

    A.fill_random();
    B.fill_random();

    auto start_time = std::chrono::high_resolution_clock::now();

    _gemm_nkm_simple_launcher<n, k, m>(A, B, C);

    std::chrono::duration<double, std::milli> ms = std::chrono::high_resolution_clock::now() - start_time;
    cout << "GPU multiplication duration: ~" << ms << "\n";

    A.cpu();
    B.cpu();
    C.cpu();

    // cout << A << '\n';
    // cout << B << '\n';
    // cout << C << '\n';

    start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            fp32 sum = 0.0;
            for (size_t r = 0; r < k; r++) {
                sum += A.get(i, r) * B.get(r, j);
            }

            // cout << sum << ' ' << C.get(i, j) << " (" << i << ", " << j << ")\n";
            // assert(std::abs(sum - C.get(i, j)) < 1e-4);
        }
    }

    ms = std::chrono::high_resolution_clock::now() - start_time;
    cout << "CPU multiplication duration: ~" << ms << "\n";
}


signed main() {

    test2();
    // auto test = Matrix<fp32>((size_t)8, (size_t)8, ROW_WISE, CPU);
    // test.cuda();

    // test.fill_random();
    // test.cpu();

    // cout << test << '\n';

    // test.to_layout(COL_WIZE);

    // cout << test;

    return 0;
}
