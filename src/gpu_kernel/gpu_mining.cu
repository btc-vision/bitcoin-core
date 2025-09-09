// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "gpu_mining.h"
#include "gpu_hash.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

extern "C" {
void LogGPUDebug(const char* message);
void LogGPUInfo(const char* message);
void LogGPUInfoFormatted(const char* format, int value);
}

// Optimized Bitcoin mining kernel with early exit on solution found
__global__ void bitcoinMiningKernel(const uint8_t* block_header, 
                                    uint32_t start_nonce,
                                    uint32_t nonce_range,
                                    const uint8_t* target,
                                    uint32_t* found_nonce,
                                    uint8_t* found_hash,
                                    volatile int* solution_found) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Early exit if another thread found a solution
    if (*solution_found) return;
    
    if (tid >= nonce_range) return;
    
    uint32_t nonce = start_nonce + tid;
    
    // Local copy of block header for this thread
    uint8_t local_header[80];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        local_header[i] = block_header[i];
    }
    
    // Insert nonce (little-endian at bytes 76-79)
    local_header[76] = (nonce >> 0) & 0xFF;
    local_header[77] = (nonce >> 8) & 0xFF;
    local_header[78] = (nonce >> 16) & 0xFF;
    local_header[79] = (nonce >> 24) & 0xFF;
    
    // Compute double SHA-256
    uint8_t hash[32];
    gpu::sha256d(local_header, 80, hash);
    
    // Check if hash meets target (compare in reverse for little-endian)
    bool valid = true;
    #pragma unroll
    for (int i = 31; i >= 0; i--) {
        if (hash[i] < target[i]) {
            break;  // Hash is less than target, valid solution
        } else if (hash[i] > target[i]) {
            valid = false;
            break;  // Hash exceeds target
        }
        // If equal, continue checking next byte
    }
    
    if (valid) {
        // Use atomicCAS to ensure only one thread writes the solution
        int old = atomicCAS((int*)solution_found, 0, 1);
        if (old == 0) {
            // This thread won the race to write the solution
            *found_nonce = nonce;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                found_hash[i] = hash[i];
            }
        }
    }
}

// Multi-GPU mining kernel for parallel search across devices
__global__ void multiGPUMiningKernel(const uint8_t* block_header,
                                     uint32_t device_id,
                                     uint32_t total_devices,
                                     uint32_t base_nonce,
                                     uint32_t total_range,
                                     const uint8_t* target,
                                     uint32_t* found_nonce,
                                     uint8_t* found_hash,
                                     volatile int* solution_found) {
    
    // Distribute work across multiple GPUs
    uint32_t range_per_device = total_range / total_devices;
    uint32_t device_start = base_nonce + (device_id * range_per_device);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    
    // Each thread processes multiple nonces for better efficiency
    for (uint32_t offset = tid; offset < range_per_device && !(*solution_found); offset += total_threads) {
        uint32_t nonce = device_start + offset;
        
        uint8_t local_header[80];
        #pragma unroll
        for (int i = 0; i < 80; i++) {
            local_header[i] = block_header[i];
        }
        
        local_header[76] = (nonce >> 0) & 0xFF;
        local_header[77] = (nonce >> 8) & 0xFF;
        local_header[78] = (nonce >> 16) & 0xFF;
        local_header[79] = (nonce >> 24) & 0xFF;
        
        uint8_t hash[32];
        gpu::sha256d(local_header, 80, hash);
        
        bool valid = true;
        for (int i = 31; i >= 0; i--) {
            if (hash[i] < target[i]) break;
            if (hash[i] > target[i]) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            int old = atomicCAS((int*)solution_found, 0, 1);
            if (old == 0) {
                *found_nonce = nonce;
                for (int i = 0; i < 32; i++) {
                    found_hash[i] = hash[i];
                }
            }
            break;
        }
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
            char msg[512];
            snprintf(msg, sizeof(msg), 
                    "GPU %d: %s (Compute %d.%d, %d SMs, %d MB)\n",
                    i, prop->name, prop->major, prop->minor,
                    prop->multiProcessorCount, 
                    (int)(prop->totalGlobalMem / (1024*1024)));
            LogGPUInfo(msg);
            m_device_properties.push_back(std::move(prop));
        } else {
            m_device_properties.push_back(nullptr);
        }
    }

    m_initialized = true;
    char msg[256];
    snprintf(msg, sizeof(msg), "GPU Mining initialized with %d device(s)\n", m_device_count);
    LogGPUInfo(msg);
    return true;
}

void GPUMiningKernel::Cleanup() {
    if (m_initialized) {
        cudaDeviceReset();
        m_initialized = false;
        m_device_properties.clear();
        LogGPUInfo("GPU Mining cleaned up\n");
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

bool GPUMiningKernel::Mine(const uint8_t* block_header, const uint8_t* target,
                           uint32_t start_nonce, uint32_t nonce_range,
                           uint32_t* found_nonce, uint8_t* found_hash) {
    if (!m_initialized || !block_header || !target || !found_nonce || !found_hash) {
        return false;
    }

    // Allocate device memory
    uint8_t* d_header = nullptr;
    uint8_t* d_target = nullptr;
    uint32_t* d_found_nonce = nullptr;
    uint8_t* d_found_hash = nullptr;
    int* d_solution_found = nullptr;
    bool success = false;
    char msg[256];
    int blockSize;
    int gridSize;
    cudaError_t err;
    int solution_found;

    if (!AllocateGPUMemory((void**)&d_header, 80) ||
        !AllocateGPUMemory((void**)&d_target, 32) ||
        !AllocateGPUMemory((void**)&d_found_nonce, sizeof(uint32_t)) ||
        !AllocateGPUMemory((void**)&d_found_hash, 32) ||
        !AllocateGPUMemory((void**)&d_solution_found, sizeof(int))) {
        goto cleanup;
    }

    // Initialize solution_found flag to 0
    cudaMemset(d_solution_found, 0, sizeof(int));

    // Copy data to GPU
    if (!CopyToGPU(d_header, block_header, 80) ||
        !CopyToGPU(d_target, target, 32)) {
        goto cleanup;
    }

    // Calculate optimal grid and block sizes
    blockSize = 256;  // Good balance for most GPUs
    gridSize = (nonce_range + blockSize - 1) / blockSize;
    
    // Limit grid size to prevent excessive resource usage
    gridSize = std::min(gridSize, 65536);

    snprintf(msg, sizeof(msg), "Mining with grid=%d, block=%d for %u nonces\n", 
             gridSize, blockSize, nonce_range);
    LogGPUDebug(msg);

    // Launch mining kernel
    bitcoinMiningKernel<<<gridSize, blockSize>>>(
        d_header, start_nonce, nonce_range, d_target,
        d_found_nonce, d_found_hash, d_solution_found
    );

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(msg, sizeof(msg), "Mining kernel launch failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        goto cleanup;
    }

    // Wait for completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        snprintf(msg, sizeof(msg), "Mining kernel sync failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        goto cleanup;
    }

    // Check if solution was found
    solution_found = 0;
    CopyFromGPU(&solution_found, d_solution_found, sizeof(int));

    if (solution_found) {
        // Copy solution back to host
        CopyFromGPU(found_nonce, d_found_nonce, sizeof(uint32_t));
        CopyFromGPU(found_hash, d_found_hash, 32);
        success = true;
        
        snprintf(msg, sizeof(msg), "Solution found! Nonce: %u\n", *found_nonce);
        LogGPUInfo(msg);
    }

cleanup:
    FreeGPUMemory(d_header);
    FreeGPUMemory(d_target);
    FreeGPUMemory(d_found_nonce);
    FreeGPUMemory(d_found_hash);
    FreeGPUMemory(d_solution_found);

    return success;
}

bool GPUMiningKernel::MineMultiGPU(const uint8_t* block_header, const uint8_t* target,
                                   uint32_t start_nonce, uint32_t total_range,
                                   uint32_t* found_nonce, uint8_t* found_hash) {
    if (!m_initialized || m_device_count < 2) {
        // Fall back to single GPU mining
        return Mine(block_header, target, start_nonce, total_range, found_nonce, found_hash);
    }

    // Distribute work across multiple GPUs
    uint32_t range_per_gpu = total_range / m_device_count;
    std::vector<std::thread> threads;
    std::atomic<bool> solution_found(false);
    std::atomic<uint32_t> best_nonce(0);
    uint8_t best_hash[32];
    std::mutex hash_mutex;
    
    for (int gpu_id = 0; gpu_id < m_device_count; gpu_id++) {
        threads.emplace_back([&, gpu_id]() {
            cudaSetDevice(gpu_id);
            
            uint32_t gpu_start = start_nonce + (gpu_id * range_per_gpu);
            uint32_t gpu_range = (gpu_id == m_device_count - 1) ? 
                                 (total_range - gpu_id * range_per_gpu) : range_per_gpu;
            
            uint32_t gpu_nonce;
            uint8_t gpu_hash[32];
            
            if (Mine(block_header, target, gpu_start, gpu_range, &gpu_nonce, gpu_hash)) {
                // Found a solution
                solution_found = true;
                std::lock_guard<std::mutex> lock(hash_mutex);
                best_nonce = gpu_nonce;
                memcpy(best_hash, gpu_hash, 32);
            }
        });
    }
    
    // Wait for all GPUs to complete
    for (auto& t : threads) {
        t.join();
    }
    
    if (solution_found) {
        *found_nonce = best_nonce;
        memcpy(found_hash, best_hash, 32);
        return true;
    }
    
    return false;
}

// Kernel to compute SHA256d hashes for testing
__global__ void testHashKernel(const uint8_t* input, uint8_t* output, int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    
    // Each thread computes SHA256d of a 64-byte block
    const uint8_t* block = input + (idx * 64);
    uint8_t* hash = output + (idx * 32);
    
    gpu::sha256d(block, 64, hash);
}

// Compatibility methods for existing interface
bool GPUMiningKernel::RunTestComputation(const int* input, int* output, int size) {
    if (!m_initialized || !input || !output || size <= 0) {
        return false;
    }
    
    // Test SHA256d computation on GPU
    // Treat input as bytes and compute SHA256d
    const int num_blocks = size / 16; // Each block is 64 bytes (16 ints)
    if (num_blocks == 0) {
        return false;
    }
    
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    
    // Allocate GPU memory
    if (!AllocateGPUMemory((void**)&d_input, num_blocks * 64) ||
        !AllocateGPUMemory((void**)&d_output, num_blocks * 32)) {
        FreeGPUMemory(d_input);
        FreeGPUMemory(d_output);
        return false;
    }
    
    // Copy input to GPU
    if (!CopyToGPU(d_input, input, num_blocks * 64)) {
        FreeGPUMemory(d_input);
        FreeGPUMemory(d_output);
        return false;
    }
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_blocks + blockSize - 1) / blockSize;
    testHashKernel<<<gridSize, blockSize>>>(d_input, d_output, num_blocks);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        FreeGPUMemory(d_input);
        FreeGPUMemory(d_output);
        return false;
    }
    
    cudaDeviceSynchronize();
    
    // Copy results back
    bool success = CopyFromGPU(output, d_output, num_blocks * 32);
    
    FreeGPUMemory(d_input);
    FreeGPUMemory(d_output);
    
    return success;
}

// Kernel to compute multiple block hashes
__global__ void computeBlockHashesKernel(const uint8_t* block_header, uint32_t start_nonce,
                                         uint32_t num_hashes, uint32_t* nonces, uint8_t* hashes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;
    
    uint32_t nonce = start_nonce + idx;
    uint8_t local_header[80];
    
    // Copy header and insert nonce
    for (int i = 0; i < 80; i++) {
        local_header[i] = block_header[i];
    }
    local_header[76] = (nonce >> 0) & 0xFF;
    local_header[77] = (nonce >> 8) & 0xFF;
    local_header[78] = (nonce >> 16) & 0xFF;
    local_header[79] = (nonce >> 24) & 0xFF;
    
    // Compute SHA256d hash
    uint8_t* hash_output = &hashes[idx * 32];
    gpu::sha256d(local_header, 80, hash_output);
    nonces[idx] = nonce;
}

bool GPUMiningKernel::ComputeBlockHashes(const uint8_t* block_header, uint32_t start_nonce,
                                         uint32_t num_hashes, uint32_t* nonces, uint8_t* hashes) {
    if (!m_initialized || !block_header || !nonces || !hashes || num_hashes == 0) {
        return false;
    }

    // Allocate GPU memory
    uint8_t* d_header = nullptr;
    uint32_t* d_nonces = nullptr;
    uint8_t* d_hashes = nullptr;
    
    if (!AllocateGPUMemory((void**)&d_header, 80) ||
        !AllocateGPUMemory((void**)&d_nonces, num_hashes * sizeof(uint32_t)) ||
        !AllocateGPUMemory((void**)&d_hashes, num_hashes * 32)) {
        FreeGPUMemory(d_header);
        FreeGPUMemory(d_nonces);
        FreeGPUMemory(d_hashes);
        return false;
    }
    
    // Copy header to GPU
    if (!CopyToGPU(d_header, block_header, 80)) {
        FreeGPUMemory(d_header);
        FreeGPUMemory(d_nonces);
        FreeGPUMemory(d_hashes);
        return false;
    }
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_hashes + blockSize - 1) / blockSize;
    computeBlockHashesKernel<<<gridSize, blockSize>>>(d_header, start_nonce, num_hashes, d_nonces, d_hashes);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        FreeGPUMemory(d_header);
        FreeGPUMemory(d_nonces);
        FreeGPUMemory(d_hashes);
        return false;
    }
    
    cudaDeviceSynchronize();
    
    // Copy results back
    bool success = CopyFromGPU(nonces, d_nonces, num_hashes * sizeof(uint32_t)) &&
                   CopyFromGPU(hashes, d_hashes, num_hashes * 32);
    
    // Clean up
    FreeGPUMemory(d_header);
    FreeGPUMemory(d_nonces);
    FreeGPUMemory(d_hashes);
    
    return success;
}

bool GPUMiningKernel::SearchForValidNonce(const uint8_t* block_header, const uint32_t* target,
                                          uint32_t start_nonce, uint32_t range,
                                          uint32_t* found_nonce, uint8_t* found_hash) {
    // Convert target from uint32_t array to byte array
    uint8_t target_bytes[32];
    for (int i = 0; i < 8; i++) {
        target_bytes[i*4 + 0] = (target[i] >> 0) & 0xFF;
        target_bytes[i*4 + 1] = (target[i] >> 8) & 0xFF;
        target_bytes[i*4 + 2] = (target[i] >> 16) & 0xFF;
        target_bytes[i*4 + 3] = (target[i] >> 24) & 0xFF;
    }
    
    return Mine(block_header, target_bytes, start_nonce, range, found_nonce, found_hash);
}