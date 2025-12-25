// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// GPU Hash Test Kernels - simple wrappers for testing GPU hash implementations
// These are test helpers, not mining code.

#include "gpu_hash.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Test Kernels - Simple wrappers to test hash functions from host
// Functions are in gpu:: namespace in gpu_hash.cuh
// ============================================================================

__global__ void SHA256TestKernel(const uint8_t* input, size_t len, uint8_t* output) {
    gpu::sha256(input, len, output);
}

__global__ void SHA256dTestKernel(const uint8_t* input, size_t len, uint8_t* output) {
    gpu::sha256d(input, len, output);
}

__global__ void SHA1TestKernel(const uint8_t* input, size_t len, uint8_t* output) {
    gpu::sha1(input, len, output);
}

__global__ void RIPEMD160TestKernel(const uint8_t* input, size_t len, uint8_t* output) {
    gpu::ripemd160(input, len, output);
}

__global__ void Hash160TestKernel(const uint8_t* input, size_t len, uint8_t* output) {
    gpu::hash160(input, len, output);
}

__global__ void SipHashTestKernel(uint64_t k0, uint64_t k1, const uint8_t* data, size_t len, uint64_t* output) {
    *output = gpu::siphash_2_4(k0, k1, data, len);
}

__global__ void MurmurHash3TestKernel(uint32_t seed, const uint8_t* data, size_t len, uint32_t* output) {
    *output = gpu::MurmurHash3(seed, data, len);
}

// ============================================================================
// C Launcher Functions (called from host code)
// ============================================================================

extern "C" {

void LaunchSHA256Test(const uint8_t* d_input, size_t len, uint8_t* d_output) {
    SHA256TestKernel<<<1, 1>>>(d_input, len, d_output);
    cudaDeviceSynchronize();
}

void LaunchSHA256dTest(const uint8_t* d_input, size_t len, uint8_t* d_output) {
    SHA256dTestKernel<<<1, 1>>>(d_input, len, d_output);
    cudaDeviceSynchronize();
}

void LaunchSHA1Test(const uint8_t* d_input, size_t len, uint8_t* d_output) {
    SHA1TestKernel<<<1, 1>>>(d_input, len, d_output);
    cudaDeviceSynchronize();
}

void LaunchRIPEMD160Test(const uint8_t* d_input, size_t len, uint8_t* d_output) {
    RIPEMD160TestKernel<<<1, 1>>>(d_input, len, d_output);
    cudaDeviceSynchronize();
}

void LaunchHash160Test(const uint8_t* d_input, size_t len, uint8_t* d_output) {
    Hash160TestKernel<<<1, 1>>>(d_input, len, d_output);
    cudaDeviceSynchronize();
}

void LaunchSipHashTest(uint64_t k0, uint64_t k1, const uint8_t* d_data, size_t len, uint64_t* d_output) {
    SipHashTestKernel<<<1, 1>>>(k0, k1, d_data, len, d_output);
    cudaDeviceSynchronize();
}

void LaunchMurmurHash3Test(uint32_t seed, const uint8_t* d_data, size_t len, uint32_t* d_output) {
    MurmurHash3TestKernel<<<1, 1>>>(seed, d_data, len, d_output);
    cudaDeviceSynchronize();
}

} // extern "C"
