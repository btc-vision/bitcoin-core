// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "gpu_mining.h"
#include "gpu_test.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <cstdio>

extern "C" {
void LogGPUDebug(const char* message);
void LogGPUInfo(const char* message);
void LogGPUInfoFormatted(const char* format, int value);
}

// Test kernel matching the demo
__global__ void miningTestKernel(int* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] = d_data[idx] * 2 + 1;
    }
}

GPUMiningKernel::GPUMiningKernel() : m_initialized(false), m_device_count(0) {
}

GPUMiningKernel::~GPUMiningKernel() {
    if (m_initialized) {
        Cleanup();
    }
}

bool GPUMiningKernel::Initialize() {
    if (m_initialized) {
        return true;
    }

    cudaError_t err = cudaGetDeviceCount(&m_device_count);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        m_device_count = 0;
        return false;
    }

    if (m_device_count == 0) {
        LogGPUInfo("No CUDA-capable GPU devices found.\n");
        return false;
    }

    // Store device properties
    m_device_properties.clear();
    for (int i = 0; i < m_device_count; i++) {
        auto prop = std::make_unique<cudaDeviceProp>();
        err = cudaGetDeviceProperties(prop.get(), i);
        if (err == cudaSuccess) {
            m_device_properties.push_back(std::move(prop));
        } else {
            m_device_properties.push_back(nullptr);
        }
    }

    m_initialized = true;
    char msg[256];
    snprintf(msg, sizeof(msg), "GPU Mining Kernel initialized with %d device(s)\n", m_device_count);
    LogGPUInfo(msg);
    return true;
}

void GPUMiningKernel::Cleanup() {
    if (m_initialized) {
        cudaDeviceReset();
        m_initialized = false;
        m_device_properties.clear();
        LogGPUInfo("GPU Mining Kernel cleaned up\n");
    }
}

int GPUMiningKernel::GetDeviceCount() const {
    return m_device_count;
}

cudaDeviceProp* GPUMiningKernel::GetDeviceProperties(int device_id) {
    if (device_id >= 0 && device_id < static_cast<int>(m_device_properties.size())) {
        return m_device_properties[device_id].get();
    }
    return nullptr;
}

bool GPUMiningKernel::AllocateGPUMemory(void** gpu_ptr, size_t size) {
    if (!m_initialized || !gpu_ptr) {
        return false;
    }

    cudaError_t err = cudaMalloc(gpu_ptr, size);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        return false;
    }
    return true;
}

void GPUMiningKernel::FreeGPUMemory(void* gpu_ptr) {
    if (gpu_ptr) {
        cudaFree(gpu_ptr);
    }
}

bool GPUMiningKernel::CopyToGPU(void* gpu_dst, const void* host_src, size_t size) {
    if (!m_initialized || !gpu_dst || !host_src) {
        return false;
    }

    cudaError_t err = cudaMemcpy(gpu_dst, host_src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        return false;
    }
    return true;
}

bool GPUMiningKernel::CopyFromGPU(void* host_dst, const void* gpu_src, size_t size) {
    if (!m_initialized || !host_dst || !gpu_src) {
        return false;
    }

    cudaError_t err = cudaMemcpy(host_dst, gpu_src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA memcpy from device failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        return false;
    }
    return true;
}

bool GPUMiningKernel::RunTestComputation(const int* input, int* output, int size) {
    if (!m_initialized || !input || !output || size <= 0) {
        return false;
    }

    const int bytes = size * sizeof(int);
    int* d_data = nullptr;

    // Allocate GPU memory
    if (!AllocateGPUMemory(reinterpret_cast<void**>(&d_data), bytes)) {
        return false;
    }

    // Copy input data to GPU
    if (!CopyToGPU(d_data, input, bytes)) {
        FreeGPUMemory(d_data);
        return false;
    }

    // Launch kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    miningTestKernel<<<gridSize, blockSize>>>(d_data, size);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        FreeGPUMemory(d_data);
        return false;
    }

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA device synchronize failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        FreeGPUMemory(d_data);
        return false;
    }

    // Copy results back to host
    bool success = CopyFromGPU(output, d_data, bytes);

    // Clean up
    FreeGPUMemory(d_data);

    return success;
}