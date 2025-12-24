#ifndef BITCOIN_GPU_KERNEL_GPU_HASH_CUH
#define BITCOIN_GPU_KERNEL_GPU_HASH_CUH

#include "gpu_types.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace gpu {

// ============================================================================
// SHA-256 Implementation
// ============================================================================

// SHA-256 constants - available on both host and device
// Note: On device, this will be placed in constant memory by the compiler when appropriate
__device__ __host__ static constexpr uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Optimized rotate right using funnel shift (with fallback for non-CUDA)
__device__ __host__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
#if defined(__CUDA_ARCH__)
    return __funnelshift_r(x, x, n);
#else
    return (x >> n) | (x << (32 - n));
#endif
}

// SHA-256 functions
__device__ __host__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __host__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __host__ __forceinline__ uint32_t Sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __host__ __forceinline__ uint32_t Sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __host__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __host__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Byte swap for endianness
__device__ __host__ __forceinline__ uint32_t bswap32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    uint32_t result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(result) : "r"(x));
    return result;
#else
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
#endif
}

// SHA-256 compression function
__device__ __host__ inline void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    
    // Prepare message schedule
    #pragma unroll 16
    for (int i = 0; i < 16; i++) {
        W[i] = bswap32(((uint32_t*)block)[i]);
    }
    
    #pragma unroll 48
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    }
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Main loop
    #pragma unroll 64
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + K256[i] + W[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    
    // Add compressed chunk to hash value
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Full SHA-256 hash function
__device__ __host__ inline void sha256(const uint8_t* data, size_t len, uint8_t* hash) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint8_t buffer[64];
    uint64_t total_len = len;
    
    // Process full blocks
    while (len >= 64) {
        sha256_transform(state, data);
        data += 64;
        len -= 64;
    }
    
    // Handle remaining bytes and padding
    memset(buffer, 0, 64);
    if (len > 0) {
        memcpy(buffer, data, len);
    }
    
    // Add padding
    buffer[len] = 0x80;
    
    // If we don't have room for the length, process this block and prepare another
    if (len >= 56) {
        sha256_transform(state, buffer);
        memset(buffer, 0, 64);
    }
    
    // Append length in bits as big-endian 64-bit integer
    uint64_t bit_len = total_len * 8;
    buffer[56] = (bit_len >> 56) & 0xFF;
    buffer[57] = (bit_len >> 48) & 0xFF;
    buffer[58] = (bit_len >> 40) & 0xFF;
    buffer[59] = (bit_len >> 32) & 0xFF;
    buffer[60] = (bit_len >> 24) & 0xFF;
    buffer[61] = (bit_len >> 16) & 0xFF;
    buffer[62] = (bit_len >> 8) & 0xFF;
    buffer[63] = bit_len & 0xFF;
    
    sha256_transform(state, buffer);
    
    // Output hash in big-endian format
    for (int i = 0; i < 8; i++) {
        hash[i * 4 + 0] = (state[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = state[i] & 0xFF;
    }
}

// Double SHA-256 (Bitcoin's proof-of-work)
__device__ __host__ inline void sha256d(const uint8_t* data, size_t len, uint8_t* hash) {
    uint8_t first_hash[32];
    sha256(data, len, first_hash);
    sha256(first_hash, 32, hash);
}

// ============================================================================
// SipHash Implementation
// ============================================================================

// SipHash round function
__device__ __forceinline__ void sipround(uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3) {
    v0 += v1; v1 = (v1 << 13) | (v1 >> 51); v1 ^= v0; v0 = (v0 << 32) | (v0 >> 32);
    v2 += v3; v3 = (v3 << 16) | (v3 >> 48); v3 ^= v2;
    v0 += v3; v3 = (v3 << 21) | (v3 >> 43); v3 ^= v0;
    v2 += v1; v1 = (v1 << 17) | (v1 >> 47); v1 ^= v2; v2 = (v2 << 32) | (v2 >> 32);
}

// SipHash-2-4 for arbitrary data
__device__ inline uint64_t siphash_2_4(uint64_t k0, uint64_t k1, const uint8_t* data, size_t len) {
    uint64_t v0 = 0x736f6d6570736575ULL ^ k0;
    uint64_t v1 = 0x646f72616e646f6dULL ^ k1;
    uint64_t v2 = 0x6c7967656e657261ULL ^ k0;
    uint64_t v3 = 0x7465646279746573ULL ^ k1;
    
    const uint8_t* end = data + (len & ~7);
    
    // Process 8-byte blocks
    while (data != end) {
        uint64_t m = ((uint64_t)data[0]) | ((uint64_t)data[1] << 8) |
                     ((uint64_t)data[2] << 16) | ((uint64_t)data[3] << 24) |
                     ((uint64_t)data[4] << 32) | ((uint64_t)data[5] << 40) |
                     ((uint64_t)data[6] << 48) | ((uint64_t)data[7] << 56);
        
        v3 ^= m;
        sipround(v0, v1, v2, v3);
        sipround(v0, v1, v2, v3);
        v0 ^= m;
        
        data += 8;
    }
    
    // Handle remaining bytes
    uint64_t last = ((uint64_t)len) << 56;
    switch (len & 7) {
        case 7: last |= ((uint64_t)data[6]) << 48;
        case 6: last |= ((uint64_t)data[5]) << 40;
        case 5: last |= ((uint64_t)data[4]) << 32;
        case 4: last |= ((uint64_t)data[3]) << 24;
        case 3: last |= ((uint64_t)data[2]) << 16;
        case 2: last |= ((uint64_t)data[1]) << 8;
        case 1: last |= ((uint64_t)data[0]);
    }
    
    v3 ^= last;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    v0 ^= last;
    
    // Finalization
    v2 ^= 0xff;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    
    return v0 ^ v1 ^ v2 ^ v3;
}

// SipHash for uint256 (optimized for Bitcoin txids)
__device__ __host__ inline uint64_t SipHashUint256(uint64_t k0, uint64_t k1, const uint256_gpu& val) {
#ifdef __CUDA_ARCH__
    // GPU version - use optimized implementation
    uint64_t v0 = 0x736f6d6570736575ULL ^ k0;
    uint64_t v1 = 0x646f72616e646f6dULL ^ k1;
    uint64_t v2 = 0x6c7967656e657261ULL ^ k0;
    uint64_t v3 = 0x7465646279746573ULL ^ k1;
    
    const uint64_t* data = reinterpret_cast<const uint64_t*>(val.data);
    
    // Process 4 words (32 bytes)
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        uint64_t m = data[i];
        v3 ^= m;
        sipround(v0, v1, v2, v3);
        sipround(v0, v1, v2, v3);
        v0 ^= m;
    }
    
    // Finalization with length
    uint64_t last = ((uint64_t)32) << 56;
    v3 ^= last;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    v0 ^= last;
    
    v2 ^= 0xff;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    
    return v0 ^ v1 ^ v2 ^ v3;
#else
    // Host version for testing
    uint64_t v0 = 0x736f6d6570736575ULL ^ k0;
    uint64_t v1 = 0x646f72616e646f6dULL ^ k1;
    uint64_t v2 = 0x6c7967656e657261ULL ^ k0;
    uint64_t v3 = 0x7465646279746573ULL ^ k1;
    
    const uint64_t* data = reinterpret_cast<const uint64_t*>(val.data);
    
    for (int i = 0; i < 4; i++) {
        uint64_t m = data[i];
        v3 ^= m;
        
        for (int j = 0; j < 2; j++) {
            v0 += v1; v1 = ((v1 << 13) | (v1 >> 51)); v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32));
            v2 += v3; v3 = ((v3 << 16) | (v3 >> 48)); v3 ^= v2;
            v0 += v3; v3 = ((v3 << 21) | (v3 >> 43)); v3 ^= v0;
            v2 += v1; v1 = ((v1 << 17) | (v1 >> 47)); v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32));
        }
        
        v0 ^= m;
    }
    
    uint64_t last = ((uint64_t)32) << 56;
    v3 ^= last;
    
    for (int j = 0; j < 2; j++) {
        v0 += v1; v1 = ((v1 << 13) | (v1 >> 51)); v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32));
        v2 += v3; v3 = ((v3 << 16) | (v3 >> 48)); v3 ^= v2;
        v0 += v3; v3 = ((v3 << 21) | (v3 >> 43)); v3 ^= v0;
        v2 += v1; v1 = ((v1 << 17) | (v1 >> 47)); v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32));
    }
    
    v0 ^= last;
    v2 ^= 0xff;
    
    for (int i = 0; i < 4; i++) {
        v0 += v1; v1 = ((v1 << 13) | (v1 >> 51)); v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32));
        v2 += v3; v3 = ((v3 << 16) | (v3 >> 48)); v3 ^= v2;
        v0 += v3; v3 = ((v3 << 21) | (v3 >> 43)); v3 ^= v0;
        v2 += v1; v1 = ((v1 << 17) | (v1 >> 47)); v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32));
    }
    
    return v0 ^ v1 ^ v2 ^ v3;
#endif
}

// SipHash with extra data (for outpoint: txid + vout)
__device__ __host__ inline uint64_t SipHashUint256Extra(uint64_t k0, uint64_t k1, 
                                                        const uint256_gpu& val, uint32_t extra) {
    // Combine txid hash with vout using proper mixing
    uint64_t base_hash = SipHashUint256(k0, k1, val);
    
    // Mix in the extra data properly
    uint8_t extra_data[4];
    memcpy(extra_data, &extra, 4);
    
#ifdef __CUDA_ARCH__
    return siphash_2_4(base_hash, base_hash >> 32, extra_data, 4);
#else
    // Host version - simple mixing
    base_hash ^= extra;
    base_hash *= 0x9e3779b97f4a7c15ULL;  // Golden ratio
    base_hash ^= (base_hash >> 33);
    base_hash *= 0xc4ceb9fe1a85ec53ULL;
    base_hash ^= (base_hash >> 33);
    return base_hash;
#endif
}

// ============================================================================
// Endian conversion helpers
// ============================================================================

// Helper function to read uint32_t in little-endian
__device__ __host__ inline uint32_t ReadLE32(const uint8_t* ptr) {
    return ((uint32_t)ptr[0]) | 
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

// Helper function to write uint32_t in little-endian
__device__ inline void WriteLE32(uint8_t* ptr, uint32_t val) {
    ptr[0] = val & 0xff;
    ptr[1] = (val >> 8) & 0xff;
    ptr[2] = (val >> 16) & 0xff;
    ptr[3] = (val >> 24) & 0xff;
}

// Helper function to write uint64_t in little-endian
__device__ inline void WriteLE64(uint8_t* ptr, uint64_t val) {
    ptr[0] = val & 0xff;
    ptr[1] = (val >> 8) & 0xff;
    ptr[2] = (val >> 16) & 0xff;
    ptr[3] = (val >> 24) & 0xff;
    ptr[4] = (val >> 32) & 0xff;
    ptr[5] = (val >> 40) & 0xff;
    ptr[6] = (val >> 48) & 0xff;
    ptr[7] = (val >> 56) & 0xff;
}

// ============================================================================
// RIPEMD-160 Implementation
// ============================================================================

// RIPEMD-160 helper functions
__device__ inline uint32_t ripemd160_f1(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
__device__ inline uint32_t ripemd160_f2(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ inline uint32_t ripemd160_f3(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
__device__ inline uint32_t ripemd160_f4(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
__device__ inline uint32_t ripemd160_f5(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

__device__ inline uint32_t ripemd160_rol(uint32_t x, int i) { 
    return (x << i) | (x >> (32 - i)); 
}

__device__ inline void ripemd160_round(uint32_t& a, uint32_t b, uint32_t& c, uint32_t d, uint32_t e, 
                                       uint32_t f, uint32_t x, uint32_t k, int r) {
    a = ripemd160_rol(a + f + x + k, r) + e;
    c = ripemd160_rol(c, 10);
}

__device__ inline void ripemd160_transform(uint32_t* s, const uint8_t* chunk) {
    uint32_t a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
    uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
    
    // Load message words
    uint32_t w[16];
    for (int i = 0; i < 16; i++) {
        w[i] = ReadLE32(chunk + i * 4);
    }
    
    // Round 1
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f1(b1, c1, d1), w[0], 0, 11);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f5(b2, c2, d2), w[5], 0x50A28BE6ul, 8);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f1(a1, b1, c1), w[1], 0, 14);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f5(a2, b2, c2), w[14], 0x50A28BE6ul, 9);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f1(e1, a1, b1), w[2], 0, 15);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f5(e2, a2, b2), w[7], 0x50A28BE6ul, 9);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f1(d1, e1, a1), w[3], 0, 12);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f5(d2, e2, a2), w[0], 0x50A28BE6ul, 11);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f1(c1, d1, e1), w[4], 0, 5);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f5(c2, d2, e2), w[9], 0x50A28BE6ul, 13);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f1(b1, c1, d1), w[5], 0, 8);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f5(b2, c2, d2), w[2], 0x50A28BE6ul, 15);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f1(a1, b1, c1), w[6], 0, 7);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f5(a2, b2, c2), w[11], 0x50A28BE6ul, 15);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f1(e1, a1, b1), w[7], 0, 9);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f5(e2, a2, b2), w[4], 0x50A28BE6ul, 5);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f1(d1, e1, a1), w[8], 0, 11);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f5(d2, e2, a2), w[13], 0x50A28BE6ul, 7);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f1(c1, d1, e1), w[9], 0, 13);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f5(c2, d2, e2), w[6], 0x50A28BE6ul, 7);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f1(b1, c1, d1), w[10], 0, 14);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f5(b2, c2, d2), w[15], 0x50A28BE6ul, 8);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f1(a1, b1, c1), w[11], 0, 15);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f5(a2, b2, c2), w[8], 0x50A28BE6ul, 11);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f1(e1, a1, b1), w[12], 0, 6);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f5(e2, a2, b2), w[1], 0x50A28BE6ul, 14);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f1(d1, e1, a1), w[13], 0, 7);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f5(d2, e2, a2), w[10], 0x50A28BE6ul, 14);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f1(c1, d1, e1), w[14], 0, 9);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f5(c2, d2, e2), w[3], 0x50A28BE6ul, 12);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f1(b1, c1, d1), w[15], 0, 8);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f5(b2, c2, d2), w[12], 0x50A28BE6ul, 6);
    
    // Round 2
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f2(a1, b1, c1), w[7], 0x5A827999ul, 7);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f4(a2, b2, c2), w[6], 0x5C4DD124ul, 9);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f2(e1, a1, b1), w[4], 0x5A827999ul, 6);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f4(e2, a2, b2), w[11], 0x5C4DD124ul, 13);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f2(d1, e1, a1), w[13], 0x5A827999ul, 8);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f4(d2, e2, a2), w[3], 0x5C4DD124ul, 15);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f2(c1, d1, e1), w[1], 0x5A827999ul, 13);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f4(c2, d2, e2), w[7], 0x5C4DD124ul, 7);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f2(b1, c1, d1), w[10], 0x5A827999ul, 11);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f4(b2, c2, d2), w[0], 0x5C4DD124ul, 12);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f2(a1, b1, c1), w[6], 0x5A827999ul, 9);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f4(a2, b2, c2), w[13], 0x5C4DD124ul, 8);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f2(e1, a1, b1), w[15], 0x5A827999ul, 7);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f4(e2, a2, b2), w[5], 0x5C4DD124ul, 9);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f2(d1, e1, a1), w[3], 0x5A827999ul, 15);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f4(d2, e2, a2), w[10], 0x5C4DD124ul, 11);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f2(c1, d1, e1), w[12], 0x5A827999ul, 7);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f4(c2, d2, e2), w[14], 0x5C4DD124ul, 7);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f2(b1, c1, d1), w[0], 0x5A827999ul, 12);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f4(b2, c2, d2), w[15], 0x5C4DD124ul, 7);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f2(a1, b1, c1), w[9], 0x5A827999ul, 15);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f4(a2, b2, c2), w[8], 0x5C4DD124ul, 12);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f2(e1, a1, b1), w[5], 0x5A827999ul, 9);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f4(e2, a2, b2), w[12], 0x5C4DD124ul, 7);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f2(d1, e1, a1), w[2], 0x5A827999ul, 11);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f4(d2, e2, a2), w[4], 0x5C4DD124ul, 6);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f2(c1, d1, e1), w[14], 0x5A827999ul, 7);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f4(c2, d2, e2), w[9], 0x5C4DD124ul, 15);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f2(b1, c1, d1), w[11], 0x5A827999ul, 13);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f4(b2, c2, d2), w[1], 0x5C4DD124ul, 13);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f2(a1, b1, c1), w[8], 0x5A827999ul, 12);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f4(a2, b2, c2), w[2], 0x5C4DD124ul, 11);
    
    // Round 3
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f3(e1, a1, b1), w[3], 0x6ED9EBA1ul, 11);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f3(e2, a2, b2), w[15], 0x6D703EF3ul, 9);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f3(d1, e1, a1), w[10], 0x6ED9EBA1ul, 13);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f3(d2, e2, a2), w[5], 0x6D703EF3ul, 7);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f3(c1, d1, e1), w[14], 0x6ED9EBA1ul, 6);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f3(c2, d2, e2), w[1], 0x6D703EF3ul, 15);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f3(b1, c1, d1), w[4], 0x6ED9EBA1ul, 7);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f3(b2, c2, d2), w[3], 0x6D703EF3ul, 11);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f3(a1, b1, c1), w[9], 0x6ED9EBA1ul, 14);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f3(a2, b2, c2), w[7], 0x6D703EF3ul, 8);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f3(e1, a1, b1), w[15], 0x6ED9EBA1ul, 9);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f3(e2, a2, b2), w[14], 0x6D703EF3ul, 6);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f3(d1, e1, a1), w[8], 0x6ED9EBA1ul, 13);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f3(d2, e2, a2), w[6], 0x6D703EF3ul, 6);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f3(c1, d1, e1), w[1], 0x6ED9EBA1ul, 15);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f3(c2, d2, e2), w[9], 0x6D703EF3ul, 14);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f3(b1, c1, d1), w[2], 0x6ED9EBA1ul, 14);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f3(b2, c2, d2), w[11], 0x6D703EF3ul, 12);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f3(a1, b1, c1), w[7], 0x6ED9EBA1ul, 8);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f3(a2, b2, c2), w[8], 0x6D703EF3ul, 13);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f3(e1, a1, b1), w[0], 0x6ED9EBA1ul, 13);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f3(e2, a2, b2), w[12], 0x6D703EF3ul, 5);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f3(d1, e1, a1), w[6], 0x6ED9EBA1ul, 6);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f3(d2, e2, a2), w[2], 0x6D703EF3ul, 14);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f3(c1, d1, e1), w[13], 0x6ED9EBA1ul, 5);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f3(c2, d2, e2), w[10], 0x6D703EF3ul, 13);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f3(b1, c1, d1), w[11], 0x6ED9EBA1ul, 12);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f3(b2, c2, d2), w[0], 0x6D703EF3ul, 13);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f3(a1, b1, c1), w[5], 0x6ED9EBA1ul, 7);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f3(a2, b2, c2), w[4], 0x6D703EF3ul, 7);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f3(e1, a1, b1), w[12], 0x6ED9EBA1ul, 5);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f3(e2, a2, b2), w[13], 0x6D703EF3ul, 5);
    
    // Round 4
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f4(d1, e1, a1), w[1], 0x8F1BBCDCul, 11);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f2(d2, e2, a2), w[8], 0x7A6D76E9ul, 15);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f4(c1, d1, e1), w[9], 0x8F1BBCDCul, 12);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f2(c2, d2, e2), w[6], 0x7A6D76E9ul, 5);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f4(b1, c1, d1), w[11], 0x8F1BBCDCul, 14);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f2(b2, c2, d2), w[4], 0x7A6D76E9ul, 8);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f4(a1, b1, c1), w[10], 0x8F1BBCDCul, 15);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f2(a2, b2, c2), w[1], 0x7A6D76E9ul, 11);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f4(e1, a1, b1), w[0], 0x8F1BBCDCul, 14);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f2(e2, a2, b2), w[3], 0x7A6D76E9ul, 14);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f4(d1, e1, a1), w[8], 0x8F1BBCDCul, 15);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f2(d2, e2, a2), w[11], 0x7A6D76E9ul, 14);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f4(c1, d1, e1), w[12], 0x8F1BBCDCul, 9);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f2(c2, d2, e2), w[15], 0x7A6D76E9ul, 6);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f4(b1, c1, d1), w[4], 0x8F1BBCDCul, 8);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f2(b2, c2, d2), w[0], 0x7A6D76E9ul, 14);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f4(a1, b1, c1), w[13], 0x8F1BBCDCul, 9);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f2(a2, b2, c2), w[5], 0x7A6D76E9ul, 6);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f4(e1, a1, b1), w[3], 0x8F1BBCDCul, 14);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f2(e2, a2, b2), w[12], 0x7A6D76E9ul, 9);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f4(d1, e1, a1), w[7], 0x8F1BBCDCul, 5);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f2(d2, e2, a2), w[2], 0x7A6D76E9ul, 12);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f4(c1, d1, e1), w[15], 0x8F1BBCDCul, 6);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f2(c2, d2, e2), w[13], 0x7A6D76E9ul, 9);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f4(b1, c1, d1), w[14], 0x8F1BBCDCul, 8);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f2(b2, c2, d2), w[9], 0x7A6D76E9ul, 12);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f4(a1, b1, c1), w[5], 0x8F1BBCDCul, 6);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f2(a2, b2, c2), w[7], 0x7A6D76E9ul, 5);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f4(e1, a1, b1), w[6], 0x8F1BBCDCul, 5);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f2(e2, a2, b2), w[10], 0x7A6D76E9ul, 15);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f4(d1, e1, a1), w[2], 0x8F1BBCDCul, 12);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f2(d2, e2, a2), w[14], 0x7A6D76E9ul, 8);
    
    // Round 5
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f5(c1, d1, e1), w[4], 0xA953FD4Eul, 9);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f1(c2, d2, e2), w[12], 0, 8);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f5(b1, c1, d1), w[0], 0xA953FD4Eul, 15);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f1(b2, c2, d2), w[15], 0, 5);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f5(a1, b1, c1), w[5], 0xA953FD4Eul, 5);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f1(a2, b2, c2), w[10], 0, 12);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f5(e1, a1, b1), w[9], 0xA953FD4Eul, 11);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f1(e2, a2, b2), w[4], 0, 9);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f5(d1, e1, a1), w[7], 0xA953FD4Eul, 6);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f1(d2, e2, a2), w[1], 0, 12);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f5(c1, d1, e1), w[12], 0xA953FD4Eul, 8);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f1(c2, d2, e2), w[5], 0, 5);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f5(b1, c1, d1), w[2], 0xA953FD4Eul, 13);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f1(b2, c2, d2), w[8], 0, 14);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f5(a1, b1, c1), w[10], 0xA953FD4Eul, 12);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f1(a2, b2, c2), w[7], 0, 6);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f5(e1, a1, b1), w[14], 0xA953FD4Eul, 5);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f1(e2, a2, b2), w[6], 0, 8);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f5(d1, e1, a1), w[1], 0xA953FD4Eul, 12);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f1(d2, e2, a2), w[2], 0, 13);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f5(c1, d1, e1), w[3], 0xA953FD4Eul, 13);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f1(c2, d2, e2), w[13], 0, 6);
    ripemd160_round(a1, b1, c1, d1, e1, ripemd160_f5(b1, c1, d1), w[8], 0xA953FD4Eul, 14);
    ripemd160_round(a2, b2, c2, d2, e2, ripemd160_f1(b2, c2, d2), w[14], 0, 5);
    ripemd160_round(e1, a1, b1, c1, d1, ripemd160_f5(a1, b1, c1), w[11], 0xA953FD4Eul, 11);
    ripemd160_round(e2, a2, b2, c2, d2, ripemd160_f1(a2, b2, c2), w[0], 0, 15);
    ripemd160_round(d1, e1, a1, b1, c1, ripemd160_f5(e1, a1, b1), w[6], 0xA953FD4Eul, 8);
    ripemd160_round(d2, e2, a2, b2, c2, ripemd160_f1(e2, a2, b2), w[3], 0, 13);
    ripemd160_round(c1, d1, e1, a1, b1, ripemd160_f5(d1, e1, a1), w[15], 0xA953FD4Eul, 5);
    ripemd160_round(c2, d2, e2, a2, b2, ripemd160_f1(d2, e2, a2), w[9], 0, 11);
    ripemd160_round(b1, c1, d1, e1, a1, ripemd160_f5(c1, d1, e1), w[13], 0xA953FD4Eul, 6);
    ripemd160_round(b2, c2, d2, e2, a2, ripemd160_f1(c2, d2, e2), w[11], 0, 11);
    
    // Combine results
    uint32_t t = s[0];
    s[0] = s[1] + c1 + d2;
    s[1] = s[2] + d1 + e2;
    s[2] = s[3] + e1 + a2;
    s[3] = s[4] + a1 + b2;
    s[4] = t + b1 + c2;
}

__device__ inline void ripemd160(const uint8_t* data, size_t len, uint8_t* hash) {
    // Initialize state
    uint32_t s[5];
    s[0] = 0x67452301ul;
    s[1] = 0xEFCDAB89ul;
    s[2] = 0x98BADCFEul;
    s[3] = 0x10325476ul;
    s[4] = 0xC3D2E1F0ul;
    
    // Process complete 64-byte chunks
    size_t bytes = 0;
    while (len >= 64) {
        ripemd160_transform(s, data);
        data += 64;
        len -= 64;
        bytes += 64;
    }
    
    // Handle remaining bytes
    uint8_t buf[64];
    memcpy(buf, data, len);
    bytes += len;
    
    // Padding
    buf[len++] = 0x80;
    if (len > 56) {
        memset(buf + len, 0, 64 - len);
        ripemd160_transform(s, buf);
        memset(buf, 0, 56);
    } else {
        memset(buf + len, 0, 56 - len);
    }
    
    // Append length
    WriteLE64(buf + 56, bytes << 3);
    ripemd160_transform(s, buf);
    
    // Write output
    WriteLE32(hash, s[0]);
    WriteLE32(hash + 4, s[1]);
    WriteLE32(hash + 8, s[2]);
    WriteLE32(hash + 12, s[3]);
    WriteLE32(hash + 16, s[4]);
}

// ============================================================================
// SHA1 Implementation
// ============================================================================

// Helper function to read uint32_t in big-endian
__device__ __host__ inline uint32_t ReadBE32(const uint8_t* ptr) {
    return ((uint32_t)ptr[0] << 24) | 
           ((uint32_t)ptr[1] << 16) |
           ((uint32_t)ptr[2] << 8) |
           ((uint32_t)ptr[3]);
}

// Helper function to write uint32_t in big-endian  
__device__ inline void WriteBE32(uint8_t* ptr, uint32_t val) {
    ptr[0] = (val >> 24) & 0xff;
    ptr[1] = (val >> 16) & 0xff;
    ptr[2] = (val >> 8) & 0xff;
    ptr[3] = val & 0xff;
}

// SHA1 helper functions
__device__ inline uint32_t sha1_f1(uint32_t b, uint32_t c, uint32_t d) { 
    return d ^ (b & (c ^ d)); 
}

__device__ inline uint32_t sha1_f2(uint32_t b, uint32_t c, uint32_t d) { 
    return b ^ c ^ d; 
}

__device__ inline uint32_t sha1_f3(uint32_t b, uint32_t c, uint32_t d) { 
    return (b & c) | (d & (b | c)); 
}

__device__ inline uint32_t sha1_left(uint32_t x) { 
    return (x << 1) | (x >> 31); 
}

__device__ inline void sha1_round(uint32_t a, uint32_t& b, uint32_t c, uint32_t d, uint32_t& e, 
                                  uint32_t f, uint32_t k, uint32_t w) {
    e += ((a << 5) | (a >> 27)) + f + k + w;
    b = (b << 30) | (b >> 2);
}

__device__ inline void sha1_transform(uint32_t* s, const uint8_t* chunk) {
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4];
    uint32_t w[16];
    
    const uint32_t k1 = 0x5A827999ul;
    const uint32_t k2 = 0x6ED9EBA1ul;
    const uint32_t k3 = 0x8F1BBCDCul;
    const uint32_t k4 = 0xCA62C1D6ul;
    
    // Load message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = ReadBE32(chunk + i * 4);
    }
    
    // Round 1 (0-19)
    for (int i = 0; i < 20; i++) {
        if (i >= 16) {
            w[i & 15] = sha1_left(w[(i+13) & 15] ^ w[(i+8) & 15] ^ w[(i+2) & 15] ^ w[i & 15]);
        }
        uint32_t f = sha1_f1(b, c, d);
        uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k1 + w[i & 15];
        e = d;
        d = c;
        c = (b << 30) | (b >> 2);
        b = a;
        a = temp;
    }
    
    // Round 2 (20-39)
    for (int i = 20; i < 40; i++) {
        w[i & 15] = sha1_left(w[(i+13) & 15] ^ w[(i+8) & 15] ^ w[(i+2) & 15] ^ w[i & 15]);
        uint32_t f = sha1_f2(b, c, d);
        uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k2 + w[i & 15];
        e = d;
        d = c;
        c = (b << 30) | (b >> 2);
        b = a;
        a = temp;
    }
    
    // Round 3 (40-59)
    for (int i = 40; i < 60; i++) {
        w[i & 15] = sha1_left(w[(i+13) & 15] ^ w[(i+8) & 15] ^ w[(i+2) & 15] ^ w[i & 15]);
        uint32_t f = sha1_f3(b, c, d);
        uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k3 + w[i & 15];
        e = d;
        d = c;
        c = (b << 30) | (b >> 2);
        b = a;
        a = temp;
    }
    
    // Round 4 (60-79)
    for (int i = 60; i < 80; i++) {
        w[i & 15] = sha1_left(w[(i+13) & 15] ^ w[(i+8) & 15] ^ w[(i+2) & 15] ^ w[i & 15]);
        uint32_t f = sha1_f2(b, c, d);
        uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k4 + w[i & 15];
        e = d;
        d = c;
        c = (b << 30) | (b >> 2);
        b = a;
        a = temp;
    }
    
    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
}

__device__ inline void sha1(const uint8_t* data, size_t len, uint8_t* hash) {
    // Initialize state
    uint32_t s[5];
    s[0] = 0x67452301ul;
    s[1] = 0xEFCDAB89ul;
    s[2] = 0x98BADCFEul;
    s[3] = 0x10325476ul;
    s[4] = 0xC3D2E1F0ul;
    
    // Process complete 64-byte chunks
    size_t bytes = 0;
    while (len >= 64) {
        sha1_transform(s, data);
        data += 64;
        len -= 64;
        bytes += 64;
    }
    
    // Handle remaining bytes
    uint8_t buf[64];
    memcpy(buf, data, len);
    bytes += len;
    
    // Padding
    buf[len++] = 0x80;
    if (len > 56) {
        memset(buf + len, 0, 64 - len);
        sha1_transform(s, buf);
        memset(buf, 0, 56);
    } else {
        memset(buf + len, 0, 56 - len);
    }
    
    // Append length in bits (big-endian)
    uint64_t bit_len = bytes << 3;
    for (int i = 0; i < 8; i++) {
        buf[56 + i] = (bit_len >> (56 - i * 8)) & 0xff;
    }
    sha1_transform(s, buf);
    
    // Write output (big-endian)
    WriteBE32(hash, s[0]);
    WriteBE32(hash + 4, s[1]);
    WriteBE32(hash + 8, s[2]);
    WriteBE32(hash + 12, s[3]);
    WriteBE32(hash + 16, s[4]);
}

// ============================================================================
// Hash160 (SHA256 + RIPEMD160) for Bitcoin addresses
// ============================================================================

__device__ inline void hash160(const uint8_t* data, size_t len, uint8_t* hash) {
    uint8_t sha_hash[32];
    sha256(data, len, sha_hash);
    ripemd160(sha_hash, 32, hash);
}

// ============================================================================
// MurmurHash3 for bloom filters (matches Bitcoin Core implementation)
// ============================================================================


__device__ __host__ inline uint32_t MurmurHash3(uint32_t seed, const uint8_t* data, size_t len) {
    // Matches Bitcoin Core's implementation exactly
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    
    const int nblocks = len / 4;
    
    // Body
    for (int i = 0; i < nblocks; i++) {
        uint32_t k1 = ReadLE32(data + i * 4);
        
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);  // rotl(k1, 15)
        k1 *= c2;
        
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);  // rotl(h1, 13)
        h1 = h1 * 5 + 0xe6546b64;
    }
    
    // Tail
    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    
    switch (len & 3) {
        case 3:
            k1 ^= ((uint32_t)tail[2]) << 16;
            // fallthrough
        case 2:
            k1 ^= ((uint32_t)tail[1]) << 8;
            // fallthrough
        case 1:
            k1 ^= ((uint32_t)tail[0]);
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);  // rotl(k1, 15)
            k1 *= c2;
            h1 ^= k1;
    }
    
    // Finalization
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    
    return h1;
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_HASH_CUH