
#include <iostream>
#include <cmath>

#include "matrix.cuh"

#include "test_class_matrix.cu"

#ifndef RUNS_NUM
#define RUNS_NUM 4
#endif

using std::cout;
using std::endl;
using std::vector;

template <typename T>
T sum_sq_diff(Matrix<T> A, Matrix<T> B) {
    A.cpu();
    B.cpu();
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    T sumA = 0.0;
    T sumB = 0.0;
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < B.cols(); j++) {
            sumA += A.get(i, j) * A.get(i, j);
            sumB += B.get(i, j) * B.get(i, j);
        }
    }
    return std::abs(sumA - sumB); 
}

template <typename T>
std::vector<std::pair<size_t, size_t> > diffs(Matrix<T> A, Matrix<T> B) {
    A.cpu();
    B.cpu();
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    vector<std::pair<size_t, size_t> > diffs;
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < B.cols(); j++) {
            if (std::abs(A.get(i, j) - B.get(i, j)) > 1e-3) {
                diffs.push_back({i, j});
            }
        }
    }
    return diffs;
}
template <typename T>
Matrix<T> mmul_cpu(Matrix<T> A, Matrix<T> B) {
    assert(A.cols() == B.rows());
    Matrix<T> C(A.rows(), B.cols(), ROW_WISE, CPU);
    for(size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < B.cols(); j++) {
            T sum = 0.0;
            for (size_t r = 0; r < A.cols(); r++) {
                sum += A.get(i, r) * B.get(r, j);
            }
            C.set(i, j, sum);
        }
    }
    return C;
}


signed test(unsigned rows1, unsigned cols1, unsigned rows2, unsigned cols2, (void) (*mul_function)(Matrix<T>, Matrix<T>, Matrix<T>), T precision = 1e-3) {
    assert(rows1 == cols2 && "геи\n");
    unsigned runs = RUNS_NUM;
        for (unsigned i = 0; i < runs; i++) {
            Matrix<fp32> A(rows1, cols1, ROW_WISE, CUDA);
            Matrix<fp32> B(rows2, cols2, ROW_WISE, CUDA);
            Matrix<fp32> G(rows1, cols2, ROW_WISE, CUDA);
    
            A.fill_random(i);
            B.fill_random(i + 1);
            mul_function(A, B, G);
            A.cpu();
            B.cpu();
            Matrix<fp32> C = mmul_cpu(A, B);
    
            auto diff = sum_sq_diff(C, G);
            if (diff > precision) {
                auto diffs_ij = diffs(C, G);
                cout << "Differing indices | diff " << endl;
                for (auto& p : diffs_ij) {
                    cout << p.first << " " << p.second  << " | " << std::abs(C.get(p.first, p.second) - G.get(p.first, p.second)) << endl;
                }
                cout << endl;
            }
        }
    return 0;
}

signed main() {
    test(128, 128, 128, 128, __gemm_nn_block_launcher);
    test(256, 128, 128, 256, __gemm_nn_block_launcher);
    test(512, 512, 512, 512, __gemm_nn_block_launcher);
    test(1024, 1024, 1024, 1024, __gemm_nn_block_launcher);
    return 0;
}