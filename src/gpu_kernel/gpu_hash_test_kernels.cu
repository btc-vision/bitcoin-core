// GPU Hash Test Kernels
#include "gpu_hash.cuh"
#include <cuda_runtime.h>

namespace gpu {

__global__ void testSHA256Kernel(const uint8_t* input, size_t len, uint8_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sha256(input, len, output);
    }
}

__global__ void testSHA256dKernel(const uint8_t* input, size_t len, uint8_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sha256d(input, len, output);
    }
}

__global__ void testSHA1Kernel(const uint8_t* input, size_t len, uint8_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sha1(input, len, output);
    }
}

__global__ void testRIPEMD160Kernel(const uint8_t* input, size_t len, uint8_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ripemd160(input, len, output);
    }
}

__global__ void testHash160Kernel(const uint8_t* input, size_t len, uint8_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        hash160(input, len, output);
    }
}

__global__ void testSipHashKernel(uint64_t k0, uint64_t k1, const uint8_t* data, size_t len, uint64_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = siphash_2_4(k0, k1, data, len);
    }
}

__global__ void testMurmurHash3Kernel(uint32_t seed, const uint8_t* data, size_t len, uint32_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = MurmurHash3(seed, data, len);
    }
}

// Wrapper functions for tests
extern "C" {

void LaunchSHA256Test(const uint8_t* input, size_t len, uint8_t* output) {
    testSHA256Kernel<<<1, 1>>>(input, len, output);
    cudaDeviceSynchronize();
}

void LaunchSHA256dTest(const uint8_t* input, size_t len, uint8_t* output) {
    testSHA256dKernel<<<1, 1>>>(input, len, output);
    cudaDeviceSynchronize();
}

void LaunchSHA1Test(const uint8_t* input, size_t len, uint8_t* output) {
    testSHA1Kernel<<<1, 1>>>(input, len, output);
    cudaDeviceSynchronize();
}

void LaunchRIPEMD160Test(const uint8_t* input, size_t len, uint8_t* output) {
    testRIPEMD160Kernel<<<1, 1>>>(input, len, output);
    cudaDeviceSynchronize();
}

void LaunchHash160Test(const uint8_t* input, size_t len, uint8_t* output) {
    testHash160Kernel<<<1, 1>>>(input, len, output);
    cudaDeviceSynchronize();
}

void LaunchSipHashTest(uint64_t k0, uint64_t k1, const uint8_t* data, size_t len, uint64_t* output) {
    testSipHashKernel<<<1, 1>>>(k0, k1, data, len, output);
    cudaDeviceSynchronize();
}

void LaunchMurmurHash3Test(uint32_t seed, const uint8_t* data, size_t len, uint32_t* output) {
    testMurmurHash3Kernel<<<1, 1>>>(seed, data, len, output);
    cudaDeviceSynchronize();
}

} // extern "C"

} // namespace gpu