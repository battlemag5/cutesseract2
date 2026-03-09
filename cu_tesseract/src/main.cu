#include "matrix.cuh"
#include <iostream>

using std::cout;
using std::endl;
using std::vector;

__global__ void set(MatrixView<fp32> view) {
    // thread coords in grid
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < view.rows && col < view.cols) {
        view(row, col) = 42;
    }
}

__global__ void mul(MatrixView<fp32> dst, MatrixView<fp32> A, MatrixView<fp32> B) {
    assert(A.cols == B.rows);
    assert(dst.rows == A.rows && dst.cols == B.cols);

    // thread coords in grid
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < dst.rows && col < dst.cols) {
        fp32 sum = 0;
        for (size_t k = 0; k < A.cols; ++k) {
            sum += A(row, k) * B(k, col);
        }
        dst(row, col) = sum;
    }
}


// void test1() {
//     Matrix<fp32> warmup(5, 5);
//     warmup.fill_random();
//     warmup.run(set, warmup.view());

//     size_t rows = 256;
//     size_t cols = 1024;
//     cout << rows << "x" << cols << endl;

//     Matrix<fp32> m(rows, cols);
//     m.fill_random();
//     cout << "GPU fill duration: " << m.run(set, m.view()) << "\n";

//     vector<fp32> host_matrix = m.download();
//     bool all_correct = true;
//     for (size_t i = 0; i < host_matrix.size(); ++i) {
//         if (host_matrix[i] != 42) {
//             cout << "Wrong value at index " << i << ": " << host_matrix[i] << endl;
//             all_correct = false;
//             break;
//         }
//     }
//     if (all_correct) {
//         cout << "filling success!" << endl;
//     }

//     Matrix<fp32> A = Matrix<fp32>(rows, cols);
//     Matrix<fp32> B = Matrix<fp32>(cols, rows);
//     Matrix<fp32> C = Matrix<fp32>(rows, rows);
//     A.fill_random(); B.fill_random();
//     cout << "GPU multiplication duration: " << C.run(mul, C.view(), A.view(), B.view()) << "\n";

//     vector<fp32> A_host = A.download();
//     vector<fp32> B_host = B.download();
//     vector<fp32> actual = C.download();
//     vector<fp32> expected(C.shape().first * C.shape().second);

//     auto start_time = std::chrono::high_resolution_clock::now();
//     for (size_t i = 0; i < C.shape().first; ++i) {
//         for (size_t j = 0; j < C.shape().second; ++j) {
//             fp32 sum = 0;
//             for (size_t k = 0; k < A.shape().second; ++k) {
//                 sum += A_host[i * A.shape().second + k] * B_host[k * C.shape().second + j];
//             }
//             expected[i * C.shape().second + j] = sum;
//         }
//     }
//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> ms = end_time - start_time;
//     cout << "CPU multiplication duration: " << ms << "\n";

//     all_correct = true;
//     for (size_t i = 0; i < actual.size(); ++i) {
//         if (std::abs(actual[i] - expected[i]) > 1e-3) {
//             cout << "Wrong value at index " << i << ": " << actual[i] << endl;
//             all_correct = false;
//             break;
//         }
//     }
//     if (all_correct) {
//         cout << "multiplication success!" << endl;
//     }
// }


signed main() {
    auto test = Matrix<fp32>((size_t)8, (size_t)8, ROW_WISE, CPU);
    test.cuda();

    test.fill_random();
    test.cpu();

    cout << test;

    return 0;
}
