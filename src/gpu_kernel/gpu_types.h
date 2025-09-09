#ifndef BITCOIN_GPU_KERNEL_GPU_TYPES_H
#define BITCOIN_GPU_KERNEL_GPU_TYPES_H

#include <cstdint>
#include <cstring>

// Define CUDA attributes for non-CUDA compilation
#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif

namespace gpu {

// GPU-compatible version of uint256
struct uint256_gpu {
    uint8_t data[32];
    
    __host__ __device__ uint256_gpu() {
        memset(data, 0, 32);
    }
    
    __host__ __device__ uint256_gpu(const uint8_t* src) {
        memcpy(data, src, 32);
    }
    
    __host__ __device__ uint256_gpu(const std::byte* src) {
        memcpy(data, src, 32);
    }
    
    __host__ __device__ bool operator==(const uint256_gpu& other) const {
        for (int i = 0; i < 32; i++) {
            if (data[i] != other.data[i]) return false;
        }
        return true;
    }
    
    __host__ __device__ bool operator!=(const uint256_gpu& other) const {
        return !(*this == other);
    }
    
    __host__ __device__ const uint8_t* begin() const {
        return data;
    }
    
    __host__ __device__ uint8_t* begin() {
        return data;
    }
};

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_TYPES_H