#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <curand.h>
#include <chrono>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <iostream>
#include <utility>

#include "dtypes.cuh"

template <typename T> class Matrix;

template <typename T>
class MatrixView {
    T* data;

    friend class Matrix<T>;
    __host__ MatrixView(size_t r, size_t c, T* d) : rows(r), cols(c), data(d) {}

public:
    size_t rows, cols;

    __device__ T& operator()(size_t row, size_t column) {
        assert(row < rows && column < cols);
        return data[row * cols + column];
    }
};

enum class DataLayout {
    ROW_WISE,
    COL_WIZE,
};

enum class DataDevice {
    CPU,
    CUDA,
};

using enum DataLayout;
using enum DataDevice;

struct Shape
{
    size_t rows;
    size_t cols;
};


template <typename T>
class Matrix {
    T* cpu_ptr;
    T* device_ptr;

    bool on_device;
    size_t rows, cols;

    DataLayout layout;
    DataDevice device;

    size_t numel;

public:
    __host__ Matrix(size_t rows, size_t cols, DataLayout layout, DataDevice device): rows(rows), cols(cols), device(device) {
        numel = sizeof(T) * rows * cols;

        if (device == CUDA) {
            cudaMalloc(&device_ptr, numel);
            cpu_ptr = nullptr;
        } else {
            cpu_ptr = new T[rows * cols];
            device_ptr = nullptr;
        }

    }

    __host__ ~Matrix() {
        if (device == CUDA) {
            cudaFree(device_ptr);
        } else {
            delete cpu_ptr;
        }
    }

    __host__ void fill_random(unsigned long long seed = 812ULL) {
        if (device == CUDA) {
            assert(sizeof(T) == sizeof(fp32)); // curandGenerateUniform is only for float32

            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, seed);
            curandGenerateUniform(gen, device_ptr, rows * cols);
            curandDestroyGenerator(gen);
        } else {
            throw std::runtime_error("Random init not implemented for cpu");
        }
    }

    __host__ void cpu() {
        if (device == CPU) return;

        device = CPU;

        cpu_ptr = new T[cols * rows];
        cudaError_t err = cudaMemcpy(cpu_ptr, device_ptr, numel, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            throw std::runtime_error("Cuda internal error");
        }

        cudaFree(device_ptr);
        device_ptr = nullptr;
    }

    __host__ void cuda() {
        if (device == CUDA) return;

        device = CUDA;

        cudaMalloc(&device_ptr, numel);
        cudaMemcpy(device_ptr, cpu_ptr, numel, cudaMemcpyHostToDevice);

        delete cpu_ptr;
        cpu_ptr = nullptr;
    }

    __host__ T* item() {
        if (device == CPU) {
            return cpu_ptr;
        } else {
            return device_ptr;
        }
    }

    __host__ friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        if (matrix.device == CUDA) {
            throw std::runtime_error("data must be on cpu for printing. consider calling .cpu()");
        }

        for (size_t i = 0; i < matrix.rows; i++) {
            os << "[";
            for (size_t j = 0; j < matrix.cols; j++) {
                os << matrix.get(i, j);
                if (j != matrix.cols - 1) os << ", ";
            }
            os << "]\n";
        }

        return os;
    }

    __host__ std::pair<size_t, size_t> shape() {
        return {rows, cols};
    }

    __host__ MatrixView<T> view() const {
        assert(device_ptr != nullptr);
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, device_ptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("Not a CUDA pointer!");
        }
        bool is_gpu = (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged);
        assert(is_gpu);

        return {rows, cols, device_ptr};
    }

    __host__ std::vector<T> download() const {
            std::vector<T> host_memory(rows * cols);
            cudaError_t err = cudaMemcpy(host_memory.data(), device_ptr, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to download matrix from device!");
            }
            return host_memory;
    }

    template <typename KernelFunction, typename... Args>
    __host__ std::chrono::duration<double, std::milli> run(KernelFunction kernel, Args... args) const {
        dim3 threads(16, 16);
        dim3 blocks((cols + threads.x - 1) / threads.x,
                    (rows + threads.y - 1) / threads.y);

        auto start_time = std::chrono::high_resolution_clock::now();
        kernel<<<blocks, threads>>>(args...);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        return end_time - start_time;
    }

    // __host__ T& operator[](size_t i, size_t j) {
    //     /* row-wise operator[] */

    //     if (i >= rows || j >= cols) {
    //         throw std::out_of_range("Index out of bounds");
    //     }
    //     if (device == CUDA) {
    //         throw std::runtime_error("data must be on cpu to get value. consider calling .cpu()");
    //     }

    //     if (layout == ROW_WISE) {
    //         return cpu_ptr[i][j];
    //     } else {
    //         return cpu_ptr[j][i];
    //     }
    // }

    __host__ T get(size_t i, size_t j) const {
        /* row-wise getter */

        if (i >= rows || j >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        if (device == CUDA) {
            throw std::runtime_error("data must be on cpu to get value. consider calling .cpu()");
        }

        if (layout == ROW_WISE) {
            return cpu_ptr[i * cols + j];
        } else {
            return cpu_ptr[i + j * rows];
        }
    }
};

#endif