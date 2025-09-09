// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_MINING_H
#define BITCOIN_GPU_KERNEL_GPU_MINING_H

#include <cstddef>
#include <memory>
#include <vector>

// Forward declarations for CUDA types
struct cudaDeviceProp;

/**
 * GPUMiningKernel class provides a high-level interface for GPU mining operations
 * Implements Bitcoin proof-of-work mining using CUDA-accelerated SHA-256d
 */
class GPUMiningKernel {
private:
    bool m_initialized;
    int m_device_count;
    std::vector<std::unique_ptr<cudaDeviceProp>> m_device_properties;

public:
    GPUMiningKernel();
    ~GPUMiningKernel();

    // Initialization and cleanup
    bool Initialize();
    void Cleanup();

    // Device information
    int GetDeviceCount() const;
    cudaDeviceProp* GetDeviceProperties(int device_id);

    // Memory management
    bool AllocateGPUMemory(void** gpu_ptr, size_t size);
    void FreeGPUMemory(void* gpu_ptr);
    bool CopyToGPU(void* gpu_dst, const void* host_src, size_t size);
    bool CopyFromGPU(void* host_dst, const void* gpu_src, size_t size);

    // Main mining interface
    bool Mine(const uint8_t* block_header, const uint8_t* target,
              uint32_t start_nonce, uint32_t nonce_range,
              uint32_t* found_nonce, uint8_t* found_hash);
    
    // Multi-GPU mining support
    bool MineMultiGPU(const uint8_t* block_header, const uint8_t* target,
                      uint32_t start_nonce, uint32_t total_range,
                      uint32_t* found_nonce, uint8_t* found_hash);

    // Compatibility methods (deprecated - use Mine() instead)
    bool RunTestComputation(const int* input, int* output, int size);
    bool ComputeBlockHashes(const uint8_t* block_header, uint32_t start_nonce, 
                            uint32_t num_hashes, uint32_t* nonces, uint8_t* hashes);
    bool SearchForValidNonce(const uint8_t* block_header, const uint32_t* target,
                             uint32_t start_nonce, uint32_t range, 
                             uint32_t* found_nonce, uint8_t* found_hash);
};

#endif // BITCOIN_GPU_KERNEL_GPU_MINING_H