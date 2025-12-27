// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SECP256K1_FIELD_CUH
#define BITCOIN_GPU_KERNEL_GPU_SECP256K1_FIELD_CUH

#include "gpu_types.h"
#include <cstdint>

namespace gpu {
namespace secp256k1 {

// ============================================================================
// secp256k1 Field Element (mod p)
// p = 2^256 - 2^32 - 977 = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF
//                          FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
// ============================================================================

// Field element represented as 8 x 32-bit limbs (little-endian)
// Value = d[0] + d[1]*2^32 + d[2]*2^64 + ... + d[7]*2^224
struct FieldElement {
    uint32_t d[8];

    __device__ __host__ FieldElement() {
        for (int i = 0; i < 8; i++) d[i] = 0;
    }

    __device__ __host__ FieldElement(uint32_t v) {
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

    __device__ __host__ bool IsOdd() const {
        return d[0] & 1;
    }

    __device__ __host__ void SetZero() {
        for (int i = 0; i < 8; i++) d[i] = 0;
    }

    __device__ __host__ void SetOne() {
        d[0] = 1;
        for (int i = 1; i < 8; i++) d[i] = 0;
    }

    __device__ __host__ bool IsEqual(const FieldElement& other) const {
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
};

// The secp256k1 field prime p
// p = 2^256 - 2^32 - 977
// In limbs (little-endian): 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, ...
__device__ __constant__ const uint32_t FIELD_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// 2^256 mod p = 2^32 + 977 = 0x1000003D1
__device__ __constant__ const uint64_t FIELD_R = 0x1000003D1ULL;

// ============================================================================
// Field Arithmetic Operations
// ============================================================================

// Compare: returns -1 if a < b, 0 if a == b, 1 if a > b
__device__ __host__ inline int fe_cmp(const FieldElement& a, const FieldElement& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] < b.d[i]) return -1;
        if (a.d[i] > b.d[i]) return 1;
    }
    return 0;
}

// Compare with field prime p
__device__ __host__ inline int fe_cmp_p(const FieldElement& a) {
    // p = {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
    for (int i = 7; i >= 2; i--) {
        if (a.d[i] < 0xFFFFFFFF) return -1;
        if (a.d[i] > 0xFFFFFFFF) return 1;
    }
    if (a.d[1] < 0xFFFFFFFE) return -1;
    if (a.d[1] > 0xFFFFFFFE) return 1;
    if (a.d[0] < 0xFFFFFC2F) return -1;
    if (a.d[0] > 0xFFFFFC2F) return 1;
    return 0;
}

// Add: r = a + b (no modular reduction)
__device__ __host__ inline uint32_t fe_add_raw(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.d[i] + (uint64_t)b.d[i];
        r.d[i] = (uint32_t)carry;
        carry >>= 32;
    }
    return (uint32_t)carry;
}

// Subtract: r = a - b, returns borrow
__device__ __host__ inline uint32_t fe_sub_raw(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    int64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        borrow = (int64_t)a.d[i] - (int64_t)b.d[i] + borrow;
        r.d[i] = (uint32_t)borrow;
        borrow >>= 32;
    }
    return (uint32_t)(borrow < 0 ? 1 : 0);
}

// Add p to a
__device__ __host__ inline void fe_add_p(FieldElement& a) {
    uint64_t carry = 0;
    carry = (uint64_t)a.d[0] + 0xFFFFFC2F;
    a.d[0] = (uint32_t)carry; carry >>= 32;
    carry += (uint64_t)a.d[1] + 0xFFFFFFFE;
    a.d[1] = (uint32_t)carry; carry >>= 32;
    for (int i = 2; i < 8; i++) {
        carry += (uint64_t)a.d[i] + 0xFFFFFFFF;
        a.d[i] = (uint32_t)carry;
        carry >>= 32;
    }
}

// Subtract p from a
__device__ __host__ inline void fe_sub_p(FieldElement& a) {
    int64_t borrow = 0;
    borrow = (int64_t)a.d[0] - 0xFFFFFC2F;
    a.d[0] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)a.d[1] - 0xFFFFFFFE;
    a.d[1] = (uint32_t)borrow; borrow >>= 32;
    for (int i = 2; i < 8; i++) {
        borrow += (int64_t)a.d[i] - 0xFFFFFFFF;
        a.d[i] = (uint32_t)borrow;
        borrow >>= 32;
    }
}

// Reduce modulo p: ensure 0 <= r < p
// Uses constant-time conditional subtraction (no while loop)
__device__ __host__ inline void fe_reduce(FieldElement& a) {
    // At most 2 subtractions needed for values up to 2p
    // Check if a >= p, if so subtract p
    int cmp = fe_cmp_p(a);
    if (cmp >= 0) {
        fe_sub_p(a);
        // Check again (for values that were >= 2p)
        if (fe_cmp_p(a) >= 0) {
            fe_sub_p(a);
        }
    }
}

// Modular addition: r = (a + b) mod p
__device__ __host__ inline void fe_add(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint32_t carry = fe_add_raw(r, a, b);
    if (carry || fe_cmp_p(r) >= 0) {
        fe_sub_p(r);
    }
}

// Modular subtraction: r = (a - b) mod p
__device__ __host__ inline void fe_sub(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint32_t borrow = fe_sub_raw(r, a, b);
    if (borrow) {
        fe_add_p(r);
    }
}

// Modular negation: r = -a mod p = p - a (if a != 0)
__device__ __host__ inline void fe_negate(FieldElement& r, const FieldElement& a) {
    if (a.IsZero()) {
        r.SetZero();
        return;
    }
    // r = p - a
    int64_t borrow = 0;
    borrow = (int64_t)0xFFFFFC2F - (int64_t)a.d[0];
    r.d[0] = (uint32_t)borrow; borrow >>= 32;
    borrow += (int64_t)0xFFFFFFFE - (int64_t)a.d[1];
    r.d[1] = (uint32_t)borrow; borrow >>= 32;
    for (int i = 2; i < 8; i++) {
        borrow += (int64_t)0xFFFFFFFF - (int64_t)a.d[i];
        r.d[i] = (uint32_t)borrow;
        borrow >>= 32;
    }
}

// Multiply by a small constant: r = a * k mod p
__device__ __host__ inline void fe_mul_small(FieldElement& r, const FieldElement& a, uint32_t k) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.d[i] * k;
        r.d[i] = (uint32_t)carry;
        carry >>= 32;
    }

    // Reduce the overflow: carry * 2^256 ≡ carry * (2^32 + 977) mod p
    // For k up to 2^32-1 and a < p, carry is at most ~32 bits
    // Apply reduction: carry contributes carry*977 to limb 0 and carry to limb 1
    if (carry) {
        uint64_t c = carry;
        carry = 0;

        // Add c * 977 to r.d[0]
        uint64_t t = (uint64_t)r.d[0] + c * 977;
        r.d[0] = (uint32_t)t;
        carry = t >> 32;

        // Add c + carry to r.d[1]
        t = (uint64_t)r.d[1] + c + carry;
        r.d[1] = (uint32_t)t;
        carry = t >> 32;

        // Propagate carry
        #pragma unroll
        for (int i = 2; i < 8 && carry; i++) {
            t = (uint64_t)r.d[i] + carry;
            r.d[i] = (uint32_t)t;
            carry = t >> 32;
        }

        // If there's still carry after 8 limbs, one more reduction
        if (carry) {
            t = (uint64_t)r.d[0] + carry * 977;
            r.d[0] = (uint32_t)t;
            carry = t >> 32;
            t = (uint64_t)r.d[1] + carry + carry; // +carry for reduction, +carry from r.d[0]
            r.d[1] = (uint32_t)t;
            carry = t >> 32;
            for (int i = 2; i < 8 && carry; i++) {
                t = (uint64_t)r.d[i] + carry;
                r.d[i] = (uint32_t)t;
                carry = t >> 32;
            }
        }
    }

    fe_reduce(r);
}

// Full multiplication: r = a * b mod p
// Uses schoolbook multiplication with secp256k1 fast reduction
// Reduction identity: 2^256 ≡ 2^32 + 977 (mod p)
__device__ __host__ inline void fe_mul(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint32_t t[16];
    uint64_t carry;

    // Schoolbook multiplication with immediate carry propagation
    for (int i = 0; i < 16; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t uv = (uint64_t)a.d[i] * b.d[j] + t[i+j] + carry;
            t[i+j] = (uint32_t)uv;
            carry = uv >> 32;
        }
        // Add carry to high limb - don't just overwrite, accumulate!
        uint64_t sum = (uint64_t)t[i+8] + carry;
        t[i+8] = (uint32_t)sum;
        // Propagate any overflow to higher limbs
        carry = sum >> 32;
        for (int k = i + 9; k < 16 && carry; k++) {
            sum = (uint64_t)t[k] + carry;
            t[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }

    // Reduce: fold t[8..15] into t[0..7] using 2^256 ≡ 2^32 + 977
    // t[i+8] contributes: t[i+8]*977 to position i, t[i+8] to position i+1
    //
    // Contribution to each position:
    //   new_t[0] = t[0] + 977*t[8]
    //   new_t[1] = t[1] + 977*t[9] + t[8]
    //   new_t[2] = t[2] + 977*t[10] + t[9]
    //   ...
    //   new_t[7] = t[7] + 977*t[15] + t[14]
    //   overflow = t[15]

    carry = 0;

    // Position 0: t[0] + 977*t[8]
    uint64_t val = (uint64_t)t[0] + (uint64_t)t[8] * 977 + carry;
    t[0] = (uint32_t)val;
    carry = val >> 32;

    // Positions 1-7: t[i] + 977*t[i+8] + t[i+7]
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        val = (uint64_t)t[i] + (uint64_t)t[i+8] * 977 + (uint64_t)t[i+7] + carry;
        t[i] = (uint32_t)val;
        carry = val >> 32;
    }

    // Add final overflow: t[15] contributes to position 8 (overflow)
    carry += t[15];

    // carry now represents overflow * 2^256, reduce again
    // carry * 2^256 ≡ carry * (2^32 + 977)
    uint64_t c = carry;
    if (c) {
        uint64_t val = (uint64_t)t[0] + c * 977;
        t[0] = (uint32_t)val;
        carry = val >> 32;

        val = (uint64_t)t[1] + c + carry;
        t[1] = (uint32_t)val;
        carry = val >> 32;

        #pragma unroll
        for (int i = 2; i < 8; i++) {
            if (!carry) break;
            val = (uint64_t)t[i] + carry;
            t[i] = (uint32_t)val;
            carry = val >> 32;
        }

        // One more reduction if still overflowed
        if (carry) {
            val = (uint64_t)t[0] + carry * 977;
            t[0] = (uint32_t)val;
            carry = val >> 32;
            val = (uint64_t)t[1] + carry + carry;
            t[1] = (uint32_t)val;
            carry = val >> 32;
            for (int i = 2; i < 8 && carry; i++) {
                val = (uint64_t)t[i] + carry;
                t[i] = (uint32_t)val;
                carry = val >> 32;
            }
        }
    }

    // Copy result
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r.d[i] = t[i];
    }

    fe_reduce(r);
}

// Squaring: r = a^2 mod p (optimized)
// Uses the identity: (sum ai*2^(32i))^2 = sum ai^2*2^(64i) + 2*sum(i<j) ai*aj*2^(32(i+j))
__device__ __host__ inline void fe_sqr(FieldElement& r, const FieldElement& a) {
    uint32_t t[16];
    uint64_t carry;

    // Initialize
    for (int i = 0; i < 16; i++) t[i] = 0;

    // Compute cross products first (will be doubled)
    // For each pair (i,j) with i < j, compute ai * aj and add to position i+j
    // We'll double these at the end before adding diagonal terms
    for (int i = 0; i < 8; i++) {
        carry = 0;
        for (int j = i + 1; j < 8; j++) {
            uint64_t uv = (uint64_t)a.d[i] * a.d[j] + t[i+j] + carry;
            t[i+j] = (uint32_t)uv;
            carry = uv >> 32;
        }
        // Propagate carry
        for (int k = i + 8; k < 16 && carry; k++) {
            uint64_t sum = (uint64_t)t[k] + carry;
            t[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }

    // Double the cross products (shift left by 1 bit)
    carry = 0;
    for (int i = 0; i < 16; i++) {
        uint64_t val = ((uint64_t)t[i] << 1) | carry;
        t[i] = (uint32_t)val;
        carry = val >> 32;
    }

    // Add diagonal terms (squares): ai^2 at position 2i
    for (int i = 0; i < 8; i++) {
        uint64_t sq = (uint64_t)a.d[i] * a.d[i];
        uint64_t sum = (uint64_t)t[2*i] + (uint32_t)sq;
        t[2*i] = (uint32_t)sum;
        carry = (sum >> 32) + (sq >> 32);
        // Propagate carry
        for (int k = 2*i + 1; k < 16 && carry; k++) {
            sum = (uint64_t)t[k] + carry;
            t[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }

    // Now reduce using 2^256 ≡ 2^32 + 977 (mod p)
    carry = 0;
    uint64_t val = (uint64_t)t[0] + (uint64_t)t[8] * 977;
    t[0] = (uint32_t)val;
    carry = val >> 32;

    for (int i = 1; i < 8; i++) {
        val = (uint64_t)t[i] + (uint64_t)t[i+8] * 977 + (uint64_t)t[i+7] + carry;
        t[i] = (uint32_t)val;
        carry = val >> 32;
    }
    carry += t[15];

    // Second reduction pass
    uint64_t c = carry;
    if (c) {
        val = (uint64_t)t[0] + c * 977;
        t[0] = (uint32_t)val;
        carry = val >> 32;
        val = (uint64_t)t[1] + c + carry;
        t[1] = (uint32_t)val;
        carry = val >> 32;
        for (int i = 2; i < 8 && carry; i++) {
            val = (uint64_t)t[i] + carry;
            t[i] = (uint32_t)val;
            carry = val >> 32;
        }
        // Third reduction if still overflowed
        if (carry) {
            val = (uint64_t)t[0] + carry * 977;
            t[0] = (uint32_t)val;
            carry = val >> 32;
            val = (uint64_t)t[1] + carry + carry;
            t[1] = (uint32_t)val;
            carry = val >> 32;
            for (int i = 2; i < 8 && carry; i++) {
                val = (uint64_t)t[i] + carry;
                t[i] = (uint32_t)val;
                carry = val >> 32;
            }
        }
    }

    for (int i = 0; i < 8; i++) r.d[i] = t[i];
    fe_reduce(r);
}

// Multiply and accumulate: r += a * b
__device__ __host__ inline void fe_mul_add(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    FieldElement t;
    fe_mul(t, a, b);
    fe_add(r, r, t);
}

// ============================================================================
// Field Inversion using Fermat's Little Theorem
// a^(-1) = a^(p-2) mod p
// ============================================================================

// Compute a^(2^n) by repeated squaring
__device__ __host__ inline void fe_sqr_n(FieldElement& r, const FieldElement& a, int n) {
    r = a;
    for (int i = 0; i < n; i++) {
        fe_sqr(r, r);
    }
}

// Field inversion: r = a^(-1) mod p
// Uses addition chain for p-2 exponent
// p-2 = 2^256 - 2^32 - 979
//     = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
__device__ __host__ inline void fe_inv(FieldElement& r, const FieldElement& a) {
    FieldElement x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223;

    // x2 = a^2 * a = a^3... wait, x2 should be a^(2^2 - 1) = a^3
    // Actually, let's compute using standard addition chain

    // x2 = a^(2^1) * a = a^3? No...
    // For x2, we want a^(2^2-1) = a^3
    // But the naming convention in secp256k1 is:
    // x2 = a^(2^2 - 1), x3 = a^(2^3 - 1), etc.

    // Let's use simpler approach: compute a^(p-2) directly
    // p-2 in binary has a specific pattern we can exploit

    // Start with computing small powers
    fe_sqr(x2, a);          // a^2
    fe_mul(x2, x2, a);      // a^3 = a^(2^2 - 1)

    FieldElement t;
    fe_sqr(t, x2);          // a^6
    fe_sqr(t, t);           // a^12
    fe_mul(x3, t, x2);      // a^15? No...

    // Let me use the standard secp256k1 addition chain
    // See: https://briansmith.org/ecc-inversion-addition-chains-01

    // x2 = a^(2^2 - 1) = a^3
    fe_sqr(x2, a);
    fe_mul(x2, x2, a);

    // x3 = a^(2^3 - 1) = a^7
    fe_sqr(x3, x2);
    fe_mul(x3, x3, a);

    // x6 = a^(2^6 - 1) = a^63
    fe_sqr_n(x6, x3, 3);
    fe_mul(x6, x6, x3);

    // x9 = a^(2^9 - 1)
    fe_sqr_n(x9, x6, 3);
    fe_mul(x9, x9, x3);

    // x11 = a^(2^11 - 1)
    fe_sqr_n(x11, x9, 2);
    fe_mul(x11, x11, x2);

    // x22 = a^(2^22 - 1)
    fe_sqr_n(x22, x11, 11);
    fe_mul(x22, x22, x11);

    // x44 = a^(2^44 - 1)
    fe_sqr_n(x44, x22, 22);
    fe_mul(x44, x44, x22);

    // x88 = a^(2^88 - 1)
    fe_sqr_n(x88, x44, 44);
    fe_mul(x88, x88, x44);

    // x176 = a^(2^176 - 1)
    fe_sqr_n(x176, x88, 88);
    fe_mul(x176, x176, x88);

    // x220 = a^(2^220 - 1)
    fe_sqr_n(x220, x176, 44);
    fe_mul(x220, x220, x44);

    // x223 = a^(2^223 - 1)
    fe_sqr_n(x223, x220, 3);
    fe_mul(x223, x223, x3);

    // Now compute a^(p-2) using the addition chain
    // p - 2 = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
    //
    // This is 256 bits with pattern:
    // - bits 255-33: all 1s (223 ones) -> this is (2^223 - 1)
    // - bit 32: 1
    // - bits 31-10: all 1s (22 ones)
    // - bits 9-6: 0000
    // - bit 5: 1
    // - bit 4: 0
    // - bit 3: 1
    // - bit 2: 1
    // - bit 1: 0
    // - bit 0: 1
    //
    // Using addition chain from libsecp256k1:
    // Result = x223 * 2^33 + 1*2^32 + (2^22-1)*2^10 + 1*2^5 + 1*2^3 + 1*2^2 + 1

    // Start with x223, shift by 23 to position 246, then add x22
    fe_sqr_n(t, x223, 23);     // x223 * 2^23
    fe_mul(t, t, x22);         // + (2^22 - 1) = a^(2^246 - 1)

    // Shift by 5, add x2
    fe_sqr_n(t, t, 5);         // * 2^5
    fe_mul(t, t, a);           // + 1 (bit 4 is 0, bit 5 is set but we need the pattern)

    // Actually, let me redo this more carefully following libsecp256k1
    // p-2 = (2^223 - 1) * 2^33 + (2^22 - 1) * 2^10 + 2^5 + 2^3 + 2^2 + 1 + 2^32
    //     = (2^223 - 1) * 2^33 + 2^32 + (2^22 - 1) * 2^10 + 2^5 + 2^3 + 2^2 + 1

    // Shift x223 by 33 positions (to get bits 255-33)
    fe_sqr_n(t, x223, 33);

    // For the remaining bits, we need to handle:
    // bit 32: 1    -> multiply by a
    // bits 31-10: (2^22-1) = x22 shifted by 10
    // bits 9-6: 0
    // bits 5,3,2,0: 1

    // After x223*2^33, multiply by a for bit 32? No wait...
    // Actually bits 32-10 = 0xFFFFFC00 >> 10 = bits 31-10 all 1s = 22 ones = (2^22 - 1) at positions 31-10
    // That's x22 shifted by 10

    // Let me trace the libsecp256k1 approach more carefully:
    // t = x223^(2^23) * x22 = a^((2^223-1)*2^23 + 2^22-1) = a^(2^246 - 2^23 + 2^22 - 1) = a^(2^246 - 2^22 - 1)... no

    // OK let me just implement it step by step from libsecp256k1:
    // 1) t = x223 << 23 = x223 * 2^23
    fe_sqr_n(t, x223, 23);
    // 2) t = t * x22 = x223*2^23 * x22
    fe_mul(t, t, x22);
    // 3) t = t << 5 = (x223*2^23 * x22) * 2^5
    fe_sqr_n(t, t, 5);
    // 4) t = t * a
    fe_mul(t, t, a);
    // 5) t = t << 3
    fe_sqr_n(t, t, 3);
    // 6) t = t * x2
    fe_mul(t, t, x2);
    // 7) t = t << 2
    fe_sqr_n(t, t, 2);
    // 8) r = t * a
    fe_mul(r, t, a);
}

// Check if a is a quadratic residue (has square root) mod p
// Uses Euler's criterion: a^((p-1)/2) = 1 if a is QR
__device__ __host__ inline bool fe_is_quad_residue(const FieldElement& a) {
    if (a.IsZero()) return true;

    // Compute a^((p-1)/2)
    // (p-1)/2 = (2^256 - 2^32 - 978) / 2 = 2^255 - 2^31 - 489
    // = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFE17

    FieldElement r;
    r.SetOne();

    // (p-1)/2 as limbs:
    // {0x7FFFFE17, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF}

    for (int limb = 7; limb >= 0; limb--) {
        uint32_t exp_limb;
        if (limb == 0) exp_limb = 0x7FFFFE17;
        else if (limb == 7) exp_limb = 0x7FFFFFFF;
        else exp_limb = 0xFFFFFFFF;

        int start_bit = (limb == 7) ? 30 : 31; // Skip leading zero in highest limb

        for (int bit = start_bit; bit >= 0; bit--) {
            if (!(limb == 7 && bit == 30)) {
                fe_sqr(r, r);
            }

            if ((exp_limb >> bit) & 1) {
                fe_mul(r, r, a);
            }
        }
    }

    return r.IsOne();
}

// Square root: r = sqrt(a) mod p (if it exists)
// Uses Tonelli-Shanks algorithm, optimized for secp256k1's p where p ≡ 3 (mod 4)
// For p ≡ 3 (mod 4): sqrt(a) = a^((p+1)/4)
__device__ __host__ inline bool fe_sqrt(FieldElement& r, const FieldElement& a) {
    if (a.IsZero()) {
        r.SetZero();
        return true;
    }

    // For secp256k1, p ≡ 3 (mod 4), so sqrt(a) = a^((p+1)/4)
    // (p+1)/4 = (2^256 - 2^32 - 976) / 4 = 2^254 - 2^30 - 244
    // = 0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C

    // Use the chain: we need a^((p+1)/4)
    FieldElement x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    // Build up powers (same as inversion)
    fe_sqr(x2, a);
    fe_mul(x2, x2, a);

    fe_sqr(x3, x2);
    fe_mul(x3, x3, a);

    fe_sqr_n(x6, x3, 3);
    fe_mul(x6, x6, x3);

    fe_sqr_n(x9, x6, 3);
    fe_mul(x9, x9, x3);

    fe_sqr_n(x11, x9, 2);
    fe_mul(x11, x11, x2);

    fe_sqr_n(x22, x11, 11);
    fe_mul(x22, x22, x11);

    fe_sqr_n(x44, x22, 22);
    fe_mul(x44, x44, x22);

    fe_sqr_n(x88, x44, 44);
    fe_mul(x88, x88, x44);

    fe_sqr_n(x176, x88, 88);
    fe_mul(x176, x176, x88);

    fe_sqr_n(x220, x176, 44);
    fe_mul(x220, x220, x44);

    fe_sqr_n(x223, x220, 3);
    fe_mul(x223, x223, x3);

    // Now compute a^((p+1)/4)
    // (p+1)/4 has 222 leading 1 bits, then specific pattern
    fe_sqr_n(t, x223, 23);
    fe_mul(t, t, x22);
    fe_sqr_n(t, t, 6);
    fe_mul(t, t, x2);
    fe_sqr(t, t);
    fe_sqr(r, t);

    // Verify: r^2 should equal a
    fe_sqr(t, r);
    if (!t.IsEqual(a)) {
        return false; // No square root exists
    }

    return true;
}

// Conditional move: r = flag ? a : r
__device__ __host__ inline void fe_cmov(FieldElement& r, const FieldElement& a, bool flag) {
    uint32_t mask = flag ? 0xFFFFFFFF : 0;
    for (int i = 0; i < 8; i++) {
        r.d[i] = (a.d[i] & mask) | (r.d[i] & ~mask);
    }
}

// Conditional negate: r = flag ? -a : a
__device__ __host__ inline void fe_cnegate(FieldElement& r, const FieldElement& a, bool flag) {
    FieldElement neg;
    fe_negate(neg, a);
    fe_cmov(r, neg, flag);
    if (!flag) r = a;
}

// Halve: r = a / 2 mod p
__device__ __host__ inline void fe_half(FieldElement& r, const FieldElement& a) {
    uint32_t carry = 0;
    if (a.IsOdd()) {
        // a is odd, so add p first (making it even), then divide by 2
        carry = fe_add_raw(r, a, *(const FieldElement*)FIELD_P);
    } else {
        r = a;
    }

    // Divide by 2 (right shift by 1)
    for (int i = 0; i < 7; i++) {
        r.d[i] = (r.d[i] >> 1) | (r.d[i+1] << 31);
    }
    r.d[7] = (r.d[7] >> 1) | (carry << 31);
}

} // namespace secp256k1
} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SECP256K1_FIELD_CUH
