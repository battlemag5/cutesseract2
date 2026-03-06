#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <curand.h>
#include <chrono>
#include <cstddef>
#include <random>
#include <stdexcept>

#include "types.cuh"

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

template <typename T>
class Matrix {
    T* data;

public:
    size_t rows, cols;
    __host__ Matrix(size_t rows, size_t cols): rows(rows), cols(cols) {
        cudaMalloc(&data, sizeof(T) * rows * cols);
    }

    __host__ ~Matrix() {
        cudaFree(data);
    }

    __host__ void fill_random(unsigned long long seed = 812ULL) {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandGenerateUniform(gen, data, rows * cols);
        curandDestroyGenerator(gen);
    }

    __host__ MatrixView<T> view() const {
        assert(data != nullptr);
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, data);
        if (err != cudaSuccess) {
            throw std::runtime_error("Not a CUDA pointer!");
        }
        bool is_gpu = (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged);
        assert(is_gpu);

        return {rows, cols, data};
    }

    __host__ std::vector<T> download() const {
            std::vector<T> host_memory(rows * cols);
            cudaError_t err = cudaMemcpy(host_memory.data(), data, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
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

    __host__ void set(size_t row, size_t col, T value) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        cudaMemcpy(data + row * cols + col, &value, sizeof(T), cudaMemcpyHostToDevice);
    }

    __host__ T get(size_t row, size_t col) const {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        T value;
        cudaMemcpy(&value, data + row * cols + col, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }
};

#endif