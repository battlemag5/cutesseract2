#pragma once

#include <cuda_runtime.h>
#include <string.h>
#include <stdexcept>

inline void cudaCheckCall(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::string err_msg = std::string(cudaGetErrorString(code)) + " in " + file + ":" + std::to_string(line);
        throw std::runtime_error(err_msg);
    }
}

#define CUDA_CHECK(call) cudaCheckCall(call, __FILE__, __LINE__)
