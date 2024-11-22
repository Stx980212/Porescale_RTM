#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
    }
}