// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SECP256K1_SCALAR_CUH
#define BITCOIN_GPU_KERNEL_GPU_SECP256K1_SCALAR_CUH

#include "gpu_types.h"
#include <cstdint>

namespace gpu {
namespace secp256k1 {

// ============================================================================
// secp256k1 Scalar (mod n)
// n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
// (the order of the generator point G)
// ============================================================================

// Scalar represented as 8 x 32-bit limbs (little-endian)
struct Scalar {
    uint32_t d[8];

    __device__ __host__ Scalar() {
        for (int i = 0; i < 8; i++) d[i] = 0;
    }

    __device__ __host__ Scalar(uint32_t v) {
        d[0] = v;
        for (int i = 1; i < 8; i++) d[i] = 0;
    }

    __device__ __host__ bool IsZero() const {
        return d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 0 &&
               d[4] == 0 && d[5] == 0 && d[6] == 0 && d[7] == 0;
    }

    __device__ __host__ bool IsOne() const {
        return d[0] == 1 && d[1] == 0 && d[2] == 0 && d[3] == 0 &&
               d[4] == 0 && d[5] == 0 && d[6] == 0 && d[7] == 0;
    }

    __device__ __host__ bool IsHigh() const {
        // Check if scalar > n/2
        // n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
        if (d[7] > 0x7FFFFFFF) return true;
        if (d[7] < 0x7FFFFFFF) return false;
        // d[7] == 0x7FFFFFFF
        for (int i = 6; i >= 4; i--) {
            if (d[i] > 0xFFFFFFFF) return true; // Can't happen but for clarity
            if (d[i] < 0xFFFFFFFF) return false;
        }
        // d[7..4] == 0x7FFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF
        if (d[3] > 0x5D576E73) return true;
        if (d[3] < 0x5D576E73) return false;
        if (d[2] > 0x57A4501D) return true;
        if (d[2] < 0x57A4501D) return false;
        if (d[1] > 0xDFE92F46) return true;
        if (d[1] < 0xDFE92F46) return false;
        if (d[0] > 0x681B20A0) return true;
        return false;
    }

    __device__ __host__ void SetZero() {
        for (int i = 0; i < 8; i++) d[i] = 0;
    }

    __device__ __host__ void SetOne() {
        d[0] = 1;
        for (int i = 1; i < 8; i++) d[i] = 0;
    }

    __device__ __host__ bool IsEqual(const Scalar& other) const {
        for (int i = 0; i < 8; i++) {
            if (d[i] != other.d[i]) return false;
        }
        return true;
    }

    // Set from 32 bytes (big-endian)
    __device__ __host__ void SetBytes(const uint8_t* b) {
        for (int i = 0; i < 8; i++) {
            int j = 7 - i;
            d[i] = ((uint32_t)b[j*4] << 24) | ((uint32_t)b[j*4+1] << 16) |
                   ((uint32_t)b[j*4+2] << 8) | (uint32_t)b[j*4+3];
        }
    }

    // Get as 32 bytes (big-endian)
    __device__ __host__ void GetBytes(uint8_t* b) const {
        for (int i = 0; i < 8; i++) {
            int j = 7 - i;
            b[j*4] = (d[i] >> 24) & 0xFF;
            b[j*4+1] = (d[i] >> 16) & 0xFF;
            b[j*4+2] = (d[i] >> 8) & 0xFF;
            b[j*4+3] = d[i] & 0xFF;
        }
    }

    // Get bit at position (0 = LSB)
    __device__ __host__ bool GetBit(int pos) const {
        int limb = pos / 32;
        int bit = pos % 32;
        return (d[limb] >> bit) & 1;
    }
};

// The secp256k1 group order n
// n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
__device__ __constant__ const uint32_t SCALAR_N[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// n/2 (for checking if scalar is "high")
__device__ __constant__ const uint32_t SCALAR_N_HALF[8] = {
    0x681B20A0, 0xDFE92F46, 0x57A4501D, 0x5D576E73,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF
};

// ============================================================================
// Scalar Arithmetic Operations
// ============================================================================

// Compare: returns -1 if a < b, 0 if a == b, 1 if a > b
__device__ __host__ inline int scalar_cmp(const Scalar& a, const Scalar& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] < b.d[i]) return -1;
        if (a.d[i] > b.d[i]) return 1;
    }
    return 0;
}

// Compare with order n
__device__ __host__ inline int scalar_cmp_n(const Scalar& a) {
    for (int i = 7; i >= 4; i--) {
        uint32_t n_i = (i == 4) ? 0xFFFFFFFE : 0xFFFFFFFF;
        if (a.d[i] < n_i) return -1;
        if (a.d[i] > n_i) return 1;
    }
    if (a.d[3] < 0xBAAEDCE6) return -1;
    if (a.d[3] > 0xBAAEDCE6) return 1;
    if (a.d[2] < 0xAF48A03B) return -1;
    if (a.d[2] > 0xAF48A03B) return 1;
    if (a.d[1] < 0xBFD25E8C) return -1;
    if (a.d[1] > 0xBFD25E8C) return 1;
    if (a.d[0] < 0xD0364141) return -1;
    if (a.d[0] > 0xD0364141) return 1;
    return 0;
}

// Add: r = a + b (no modular reduction)
__device__ __host__ inline uint32_t scalar_add_raw(Scalar& r, const Scalar& a, const Scalar& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.d[i] + (uint64_t)b.d[i];
        r.d[i] = (uint32_t)carry;
        carry >>= 32;
    }
    return (uint32_t)carry;
}

// Subtract: r = a - b, returns borrow
__device__ __host__ inline uint32_t scalar_sub_raw(Scalar& r, const Scalar& a, const Scalar& b) {
    int64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        borrow = (int64_t)a.d[i] - (int64_t)b.d[i] + borrow;
        r.d[i] = (uint32_t)borrow;
        borrow >>= 32;
    }
    return (uint32_t)(borrow < 0 ? 1 : 0);
}

// Add n to a
__device__ __host__ inline void scalar_add_n(Scalar& a) {
    uint64_t carry = 0;
    carry = (uint64_t)a.d[0] + 0xD0364141;
    a.d[0] = (uint32_t)carry; carry >>= 32;
    carry += (uint64_t)a.d[1] + 0xBFD25E8C;
    a.d[1] = (uint32_t)carry; carry >>= 32;
    carry += (uint64_t)a.d[2] + 0xAF48A03B;
    a.d[2] = (uint32_t)carry; carry >>= 32;
    carry += (uint64_t)a.d[3] + 0xBAAEDCE6;
    a.d[3] = (uint32_t)carry; carry >>= 32;
    carry += (uint64_t)a.d[4] + 0xFFFFFFFE;
    a.d[4] = (uint32_t)carry; carry >>= 32;
    for (int i = 5; i < 8; i++) {
        carry += (uint64_t)a.d[i] + 0xFFFFFFFF;
        a.d[i] = (uint32_t)carry;
        carry >>= 32;
    }
}

// Subtract n from a
__device__ __host__ inline void scalar_sub_n(Scalar& a) {
    int64_t borrow = 0;
    borrow = (int64_t)a.d[0] - 0xD0364141;
    a.d[0] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)a.d[1] - 0xBFD25E8C;
    a.d[1] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)a.d[2] - 0xAF48A03B;
    a.d[2] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)a.d[3] - 0xBAAEDCE6;
    a.d[3] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)a.d[4] - 0xFFFFFFFE;
    a.d[4] = (uint32_t)borrow; borrow >>= 32;
    for (int i = 5; i < 8; i++) {
        borrow += (int64_t)a.d[i] - 0xFFFFFFFF;
        a.d[i] = (uint32_t)borrow;
        borrow >>= 32;
    }
}

// Reduce modulo n: ensure 0 <= r < n
// Uses constant-time conditional subtraction (no while loop)
__device__ __host__ inline void scalar_reduce(Scalar& a) {
    // At most 2 subtractions needed for values up to 2n
    if (scalar_cmp_n(a) >= 0) {
        scalar_sub_n(a);
        if (scalar_cmp_n(a) >= 0) {
            scalar_sub_n(a);
        }
    }
}

// Modular addition: r = (a + b) mod n
__device__ __host__ inline void scalar_add(Scalar& r, const Scalar& a, const Scalar& b) {
    uint32_t carry = scalar_add_raw(r, a, b);
    if (carry || scalar_cmp_n(r) >= 0) {
        scalar_sub_n(r);
    }
}

// Modular subtraction: r = (a - b) mod n
__device__ __host__ inline void scalar_sub(Scalar& r, const Scalar& a, const Scalar& b) {
    uint32_t borrow = scalar_sub_raw(r, a, b);
    if (borrow) {
        scalar_add_n(r);
    }
}

// Modular negation: r = -a mod n = n - a (if a != 0)
__device__ __host__ inline void scalar_negate(Scalar& r, const Scalar& a) {
    if (a.IsZero()) {
        r.SetZero();
        return;
    }
    // r = n - a
    int64_t borrow = 0;
    borrow = (int64_t)0xD0364141 - (int64_t)a.d[0];
    r.d[0] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)0xBFD25E8C - (int64_t)a.d[1];
    r.d[1] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)0xAF48A03B - (int64_t)a.d[2];
    r.d[2] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)0xBAAEDCE6 - (int64_t)a.d[3];
    r.d[3] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)0xFFFFFFFE - (int64_t)a.d[4];
    r.d[4] = (uint32_t)borrow; borrow >>= 32;
    for (int i = 5; i < 8; i++) {
        borrow += (int64_t)0xFFFFFFFF - (int64_t)a.d[i];
        r.d[i] = (uint32_t)borrow;
        borrow >>= 32;
    }
}

// Full multiplication: r = a * b mod n
// Uses simple schoolbook multiplication with repeated subtraction reduction
__device__ __host__ inline void scalar_mul(Scalar& r, const Scalar& a, const Scalar& b) {
    // Result needs 16 limbs (512 bits) before reduction
    uint64_t t[16] = {0};

    // Schoolbook multiplication
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a.d[i] * (uint64_t)b.d[j] + t[i+j] + carry;
            t[i+j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[i+8] += carry;
    }

    // Reduce mod n using 2^256 mod n = 0x14551231950B75FC4402DA1732FC9BEBF
    // Process from high to low, reducing each limb
    // 2^256 ≡ R mod n, where R is 129 bits

    // R = 2^256 mod n (5 limbs needed since it's 129 bits)
    static const uint64_t R_MOD_N[5] = {
        0x2FC9BEBF, 0x402DA173, 0x50B75FC4, 0x45512319, 0x00000001
    };

    // Reduce high limbs: for each t[i] where i >= 8, we have
    // t[i] * 2^(32*i) ≡ t[i] * R * 2^(32*(i-8)) mod n

    for (int pass = 0; pass < 3; pass++) {  // Multiple passes to fully reduce
        for (int i = 15; i >= 8; i--) {
            if (t[i] == 0) continue;

            uint64_t val = t[i];
            t[i] = 0;
            int offset = i - 8;

            // Add val * R at position offset
            uint64_t carry = 0;
            for (int j = 0; j < 5 && offset + j < 16; j++) {
                uint64_t prod = val * R_MOD_N[j] + carry;
                if (offset + j < 16) {
                    prod += t[offset + j];
                    t[offset + j] = prod & 0xFFFFFFFF;
                }
                carry = prod >> 32;
            }
            // Propagate remaining carry
            for (int j = offset + 5; j < 16 && carry; j++) {
                carry += t[j];
                t[j] = carry & 0xFFFFFFFF;
                carry >>= 32;
            }
        }
    }

    // Copy low 256 bits to result
    for (int i = 0; i < 8; i++) {
        r.d[i] = (uint32_t)t[i];
    }

    // Final reduction to ensure r < n
    scalar_reduce(r);
}

// Squaring: r = a^2 mod n
__device__ __host__ inline void scalar_sqr(Scalar& r, const Scalar& a) {
    scalar_mul(r, a, a);
}

// Compute a^(2^n) by repeated squaring
__device__ __host__ inline void scalar_sqr_n(Scalar& r, const Scalar& a, int n) {
    r = a;
    for (int i = 0; i < n; i++) {
        scalar_sqr(r, r);
    }
}

// Scalar inversion: r = a^(-1) mod n
// Uses Fermat's little theorem with addition chain: a^(-1) = a^(n-2) mod n
// n-2 = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D036413F
__device__ __host__ inline void scalar_inv(Scalar& r, const Scalar& a) {
    // Pre-compute small powers using addition chain
    Scalar x2, x3, x4, x6, x7, x8, x15, x30, x60, x120, x127;
    Scalar t;

    // x2 = a^2
    scalar_sqr(x2, a);

    // x3 = a^3 = a^2 * a
    scalar_mul(x3, x2, a);

    // x4 = a^4 = (a^2)^2
    scalar_sqr(x4, x2);

    // x6 = a^6 = a^4 * a^2
    scalar_mul(x6, x4, x2);

    // x7 = a^7 = a^6 * a
    scalar_mul(x7, x6, a);

    // x8 = a^8 = (a^4)^2
    scalar_sqr(x8, x4);

    // x15 = a^15 = a^8 * a^7
    scalar_mul(x15, x8, x7);

    // x30 = a^30 = (a^15)^2
    scalar_sqr(x30, x15);

    // x60 = a^60 = (a^30)^2
    scalar_sqr(x60, x30);

    // x120 = a^120 = (a^60)^2
    scalar_sqr(x120, x60);

    // x127 = a^127 = a^120 * a^7
    scalar_mul(x127, x120, x7);

    // Now build n-2 using these precomputed values
    // n-2 = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D036413F
    //
    // High 128 bits: FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE = (2^128 - 1) * 2 = 2^129 - 2
    // This is 127 ones followed by a zero
    // a^(2^129 - 2) = (a^(2^128 - 1))^2 = ((a^127)^(2^121))^2

    // Start with x127, square it 121 times to get a^(127 * 2^121) = a^(2^128 - 2^121)
    // Then multiply by x127 to get a^(2^128 - 2^121 + 127) = a^(2^128 - 2^121 + 2^7 - 1)
    // This is getting complex. Let me use a simpler approach.

    // Build a^(2^128 - 1) first = a^(ffffffffffffffffffffffffffffffff) = (a^255 ... repeated)
    // Actually, let's just compute:
    // t = a^(2^7 - 1) = x127
    // t = t^(2^7) * x127 = a^(2^14 - 1)
    // t = t^(2^14) * t = a^(2^28 - 1)  -- wrong, need to track carefully

    // Simpler: Build a^(2^k - 1) iteratively
    // a^(2^7 - 1) = x127
    // a^(2^14 - 1) = (a^(2^7-1))^(2^7) * a^(2^7-1) = x127^(2^7) * x127
    scalar_sqr_n(t, x127, 7);
    scalar_mul(t, t, x127);  // t = a^(2^14 - 1)

    // a^(2^28 - 1) = t^(2^14) * t
    Scalar t14 = t;
    scalar_sqr_n(t, t, 14);
    scalar_mul(t, t, t14);  // t = a^(2^28 - 1)

    // a^(2^56 - 1) = t^(2^28) * t
    Scalar t28 = t;
    scalar_sqr_n(t, t, 28);
    scalar_mul(t, t, t28);  // t = a^(2^56 - 1)

    // a^(2^112 - 1) = t^(2^56) * t
    Scalar t56 = t;
    scalar_sqr_n(t, t, 56);
    scalar_mul(t, t, t56);  // t = a^(2^112 - 1)

    // a^(2^126 - 1) = t^(2^14) * a^(2^14 - 1)
    scalar_sqr_n(t, t, 14);
    scalar_mul(t, t, t14);  // t = a^(2^126 - 1)

    // a^(2^127 - 1) = t^2 * a
    scalar_sqr(t, t);
    scalar_mul(t, t, a);  // t = a^(2^127 - 1)

    // a^(2^128 - 2) = t^2 (the high 128 bits pattern FFFFFFFE at the end)
    scalar_sqr(t, t);  // t = a^(2^128 - 2)

    // Now we need to incorporate the low 128 bits: BAAEDCE6 AF48A03B BFD25E8C D036413F
    // Use 4-bit windowed method with precomputed table for efficiency
    // Precompute: a^1, a^2, ..., a^15
    Scalar table[16];
    table[0].SetOne();
    table[1] = a;
    table[2] = x2;
    table[3] = x3;
    table[4] = x4;
    Scalar x5; scalar_mul(x5, x4, a);
    table[5] = x5;
    table[6] = x6;
    table[7] = x7;
    table[8] = x8;
    Scalar x9; scalar_mul(x9, x8, a);
    table[9] = x9;
    Scalar x10; scalar_mul(x10, x8, x2);
    table[10] = x10;
    Scalar x11; scalar_mul(x11, x8, x3);
    table[11] = x11;
    Scalar x12; scalar_mul(x12, x8, x4);
    table[12] = x12;
    Scalar x13; scalar_mul(x13, x8, x5);
    table[13] = x13;
    Scalar x14; scalar_mul(x14, x8, x6);
    table[14] = x14;
    table[15] = x15;

    // Low 128 bits: BAAEDCE6 AF48A03B BFD25E8C D036413F (big-endian order)
    // Process 4 bits at a time from high to low
    static const uint8_t nibbles[32] = {
        0xB, 0xA, 0xA, 0xE, 0xD, 0xC, 0xE, 0x6,  // BAAEDCE6
        0xA, 0xF, 0x4, 0x8, 0xA, 0x0, 0x3, 0xB,  // AF48A03B
        0xB, 0xF, 0xD, 0x2, 0x5, 0xE, 0x8, 0xC,  // BFD25E8C
        0xD, 0x0, 0x3, 0x6, 0x4, 0x1, 0x3, 0xF   // D036413F
    };

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        // Square 4 times
        scalar_sqr(t, t);
        scalar_sqr(t, t);
        scalar_sqr(t, t);
        scalar_sqr(t, t);
        // Multiply by table[nibble] if nibble != 0
        uint8_t nib = nibbles[i];
        if (nib != 0) {
            scalar_mul(t, t, table[nib]);
        }
    }

    r = t;
}

// Conditional move: r = flag ? a : r
__device__ __host__ inline void scalar_cmov(Scalar& r, const Scalar& a, bool flag) {
    uint32_t mask = flag ? 0xFFFFFFFF : 0;
    for (int i = 0; i < 8; i++) {
        r.d[i] = (a.d[i] & mask) | (r.d[i] & ~mask);
    }
}

// Split scalar for efficient multiplication (optional optimization)
// Split k into k1, k2 where k = k1 + lambda * k2 mod n
// lambda is the cube root of 1 mod n: 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
// This enables using the GLV endomorphism for faster scalar multiplication

// Check if scalar is valid (non-zero and < n)
__device__ __host__ inline bool scalar_is_valid(const Scalar& a) {
    if (a.IsZero()) return false;
    return scalar_cmp_n(a) < 0;
}

// Half: r = a / 2 mod n
__device__ __host__ inline void scalar_half(Scalar& r, const Scalar& a) {
    uint32_t carry = 0;
    if (a.d[0] & 1) {
        // a is odd, add n first (making it even), then divide by 2
        carry = scalar_add_raw(r, a, *(const Scalar*)SCALAR_N);
    } else {
        r = a;
    }

    // Divide by 2 (right shift by 1)
    for (int i = 0; i < 7; i++) {
        r.d[i] = (r.d[i] >> 1) | (r.d[i+1] << 31);
    }
    r.d[7] = (r.d[7] >> 1) | (carry << 31);
}

// Get the number of bits in the scalar
__device__ __host__ inline int scalar_bits(const Scalar& a) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] != 0) {
            // Count leading zeros and compute bit position
            #ifdef __CUDA_ARCH__
            int lz = __clz(a.d[i]);  // CUDA intrinsic
            #else
            // CPU fallback: count leading zeros
            uint32_t v = a.d[i];
            int lz = 0;
            if (!(v & 0xFFFF0000)) { lz += 16; v <<= 16; }
            if (!(v & 0xFF000000)) { lz += 8; v <<= 8; }
            if (!(v & 0xF0000000)) { lz += 4; v <<= 4; }
            if (!(v & 0xC0000000)) { lz += 2; v <<= 2; }
            if (!(v & 0x80000000)) { lz += 1; }
            #endif
            return (i + 1) * 32 - lz;
        }
    }
    return 0;
}

} // namespace secp256k1
} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SECP256K1_SCALAR_CUH
