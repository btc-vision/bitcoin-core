#ifndef BITCOIN_GPU_KERNEL_GPU_UTILS_H
#define BITCOIN_GPU_KERNEL_GPU_UTILS_H

#include "gpu_types.h"
#include <uint256.h>

namespace gpu {

// Conversion utilities between CPU and GPU types
inline uint256_gpu ToGPU(const uint256& hash) {
    return uint256_gpu(hash.begin());
}

inline uint256 FromGPU(const uint256_gpu& gpu_hash) {
    uint256 result;
    memcpy(result.begin(), gpu_hash.data, 32);
    return result;
}

// Template to handle any type that has begin() method returning uint8_t*
template<typename T>
inline uint256_gpu ToGPU(const T& txid_like) {
    return uint256_gpu(txid_like.begin());
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_UTILS_H