// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SECP256K1_ECMULT_CUH
#define BITCOIN_GPU_KERNEL_GPU_SECP256K1_ECMULT_CUH

#include "gpu_secp256k1_group.cuh"
#include "gpu_secp256k1_scalar.cuh"
#include <cstdint>

namespace gpu {
namespace secp256k1 {

// ============================================================================
// Scalar Multiplication: r = k * P
// ============================================================================

// Simple double-and-add algorithm
// Not constant-time, suitable for public key verification
__device__ __host__ inline void ecmult_simple(JacobianPoint& r, const JacobianPoint& p, const Scalar& k) {
    if (k.IsZero() || p.IsInfinity()) {
        r.SetInfinity();
        return;
    }

    if (k.IsOne()) {
        r = p;
        return;
    }

    r.SetInfinity();

    // Find the highest set bit
    int bits = scalar_bits(k);

    // Double-and-add from high bit to low bit
    for (int i = bits - 1; i >= 0; i--) {
        // Double (use temp to avoid aliasing issues)
        if (!r.IsInfinity()) {
            JacobianPoint temp;
            point_double(temp, r);
            r = temp;
        }

        // Add if bit is set
        if (k.GetBit(i)) {
            if (r.IsInfinity()) {
                r = p;
            } else {
                JacobianPoint temp;
                point_add(temp, r, p);
                r = temp;
            }
        }
    }
}

// Window-based scalar multiplication with precomputed table
// More efficient for repeated use with the same base point
// Window size = 4 bits, table has 16 entries
static constexpr int WINDOW_SIZE = 4;
static constexpr int TABLE_SIZE = 1 << WINDOW_SIZE; // 16

struct PrecomputedTable {
    JacobianPoint table[TABLE_SIZE]; // table[i] = i * P

    __device__ __host__ void Init(const JacobianPoint& p) {
        table[0].SetInfinity();
        table[1] = p;

        // table[2] = 2*P
        point_double(table[2], p);

        // table[i] = table[i-1] + P for i >= 3
        for (int i = 3; i < TABLE_SIZE; i++) {
            point_add(table[i], table[i-1], p);
        }
    }
};

// Scalar multiplication using precomputed table
__device__ __host__ inline void ecmult_windowed(JacobianPoint& r, const PrecomputedTable& table, const Scalar& k) {
    if (k.IsZero()) {
        r.SetInfinity();
        return;
    }

    r.SetInfinity();

    // Process scalar in 4-bit windows from high to low
    int bits = scalar_bits(k);
    int top_window = (bits + WINDOW_SIZE - 1) / WINDOW_SIZE;

    for (int i = top_window - 1; i >= 0; i--) {
        // Double 4 times
        for (int j = 0; j < WINDOW_SIZE; j++) {
            if (!r.IsInfinity()) {
                point_double(r, r);
            }
        }

        // Extract 4-bit window
        int bit_pos = i * WINDOW_SIZE;
        uint32_t window = 0;
        for (int j = 0; j < WINDOW_SIZE && bit_pos + j < 256; j++) {
            if (k.GetBit(bit_pos + j)) {
                window |= (1 << j);
            }
        }

        // Add table[window]
        if (window != 0) {
            if (r.IsInfinity()) {
                r = table.table[window];
            } else {
                point_add(r, r, table.table[window]);
            }
        }
    }
}

// ============================================================================
// Generator Multiplication: r = k * G
// Uses precomputed table for the generator point
// ============================================================================

// Precomputed table for generator point G
// In practice, this would be stored in constant memory
struct GeneratorTable {
    // For each of 64 windows (256 bits / 4 bits per window),
    // we store 16 multiples of G * 16^i
    JacobianPoint tables[64][16];

    __device__ __host__ void Init() {
        AffinePoint g_affine;
        GetGenerator(g_affine);

        JacobianPoint g;
        g.FromAffine(g_affine);

        JacobianPoint base = g;

        for (int i = 0; i < 64; i++) {
            tables[i][0].SetInfinity();
            tables[i][1] = base;

            // Compute 2*base through 15*base
            point_double(tables[i][2], base);
            for (int j = 3; j < 16; j++) {
                point_add(tables[i][j], tables[i][j-1], base);
            }

            // Update base = 16 * base for next window
            for (int j = 0; j < 4; j++) {
                point_double(base, base);
            }
        }
    }
};

// Generator multiplication using precomputed table
__device__ __host__ inline void ecmult_gen(JacobianPoint& r, const GeneratorTable& gtable, const Scalar& k) {
    if (k.IsZero()) {
        r.SetInfinity();
        return;
    }

    r.SetInfinity();

    // Process each 4-bit window
    for (int i = 0; i < 64; i++) {
        int bit_pos = i * 4;
        uint32_t window = 0;
        for (int j = 0; j < 4; j++) {
            if (k.GetBit(bit_pos + j)) {
                window |= (1 << j);
            }
        }

        if (window != 0) {
            if (r.IsInfinity()) {
                r = gtable.tables[i][window];
            } else {
                point_add(r, r, gtable.tables[i][window]);
            }
        }
    }
}

// ============================================================================
// Combined Multiplication: r = k1 * G + k2 * P
// Used in ECDSA verification
// ============================================================================

__device__ __host__ inline void ecmult_multi(
    JacobianPoint& r,
    const GeneratorTable& gtable,
    const Scalar& k1,        // scalar for G
    const JacobianPoint& p,  // arbitrary point
    const Scalar& k2)        // scalar for P
{
    JacobianPoint r1, r2;

    // Compute k1 * G
    ecmult_gen(r1, gtable, k1);

    // Compute k2 * P using simple algorithm
    ecmult_simple(r2, p, k2);

    // r = r1 + r2
    point_add(r, r1, r2);
}

// ============================================================================
// Batch operations for parallel verification
// ============================================================================

// Structure for batch point operations
struct BatchPoint {
    JacobianPoint point;
    bool valid;

    __device__ __host__ BatchPoint() : valid(false) {}
};

// Batch normalize: convert multiple Jacobian points to affine efficiently
// Uses Montgomery's trick for batch inversion
__device__ __host__ inline void batch_normalize(AffinePoint* out, const JacobianPoint* in, int count) {
    if (count == 0) return;

    // Accumulate products of Z coordinates
    FieldElement* acc = new FieldElement[count];
    acc[0].SetOne();

    for (int i = 0; i < count; i++) {
        if (in[i].IsInfinity()) {
            if (i > 0) acc[i] = acc[i-1];
            else acc[i].SetOne();
        } else {
            if (i == 0) {
                acc[i] = in[i].z;
            } else {
                fe_mul(acc[i], acc[i-1], in[i].z);
            }
        }
    }

    // Compute inverse of the product
    FieldElement inv_prod;
    fe_inv(inv_prod, acc[count-1]);

    // Work backwards to compute individual inverses
    for (int i = count - 1; i >= 0; i--) {
        if (in[i].IsInfinity()) {
            out[i].SetInfinity();
        } else {
            FieldElement z_inv;

            if (i == 0) {
                z_inv = inv_prod;
            } else {
                fe_mul(z_inv, inv_prod, acc[i-1]);
                fe_mul(inv_prod, inv_prod, in[i].z);
            }

            // Convert to affine
            out[i].infinity = false;
            FieldElement z_inv2, z_inv3;
            fe_sqr(z_inv2, z_inv);
            fe_mul(z_inv3, z_inv2, z_inv);
            fe_mul(out[i].x, in[i].x, z_inv2);
            fe_mul(out[i].y, in[i].y, z_inv3);
        }
    }

    delete[] acc;
}

// ============================================================================
// Public Key Operations
// ============================================================================

// Parse a compressed public key (33 bytes)
// First byte: 0x02 (even y) or 0x03 (odd y)
// Remaining 32 bytes: x-coordinate
__device__ __host__ inline bool pubkey_parse_compressed(AffinePoint& p, const uint8_t* data) {
    if (data[0] != 0x02 && data[0] != 0x03) {
        return false;
    }

    bool y_odd = (data[0] == 0x03);

    // Parse x-coordinate
    p.x.SetBytes(data + 1);

    // Compute y^2 = x^3 + 7
    FieldElement y2, x3, seven;
    fe_sqr(x3, p.x);
    fe_mul(x3, x3, p.x);
    seven.SetZero();
    seven.d[0] = 7;
    fe_add(y2, x3, seven);

    // Compute y = sqrt(y^2)
    if (!fe_sqrt(p.y, y2)) {
        return false; // No valid y exists
    }

    // Choose the correct y based on parity
    if (p.y.IsOdd() != y_odd) {
        fe_negate(p.y, p.y);
    }

    p.infinity = false;
    return p.IsOnCurve();
}

// Parse an uncompressed public key (65 bytes)
// First byte: 0x04
// Next 32 bytes: x-coordinate
// Last 32 bytes: y-coordinate
__device__ __host__ inline bool pubkey_parse_uncompressed(AffinePoint& p, const uint8_t* data) {
    if (data[0] != 0x04) {
        return false;
    }

    p.x.SetBytes(data + 1);
    p.y.SetBytes(data + 33);
    p.infinity = false;

    return p.IsOnCurve();
}

// Parse a public key (compressed or uncompressed)
__device__ __host__ inline bool pubkey_parse(AffinePoint& p, const uint8_t* data, uint32_t len) {
    if (len == 33) {
        return pubkey_parse_compressed(p, data);
    } else if (len == 65) {
        return pubkey_parse_uncompressed(p, data);
    }
    return false;
}

// Serialize a public key as compressed (33 bytes)
__device__ __host__ inline void pubkey_serialize_compressed(uint8_t* out, const AffinePoint& p) {
    out[0] = p.y.IsOdd() ? 0x03 : 0x02;
    p.x.GetBytes(out + 1);
}

// Serialize a public key as uncompressed (65 bytes)
__device__ __host__ inline void pubkey_serialize_uncompressed(uint8_t* out, const AffinePoint& p) {
    out[0] = 0x04;
    p.x.GetBytes(out + 1);
    p.y.GetBytes(out + 33);
}

// ============================================================================
// Parse x-only public key (BIP340 Schnorr)
// ============================================================================

__device__ __host__ inline bool pubkey_parse_xonly(AffinePoint& p, const uint8_t* data) {
    // Parse x-coordinate
    p.x.SetBytes(data);

    // Compute y^2 = x^3 + 7
    FieldElement y2, x3, seven;
    fe_sqr(x3, p.x);
    fe_mul(x3, x3, p.x);
    seven.SetZero();
    seven.d[0] = 7;
    fe_add(y2, x3, seven);

    // Compute y = sqrt(y^2)
    if (!fe_sqrt(p.y, y2)) {
        return false;
    }

    // BIP340: always use the even y-coordinate
    if (p.y.IsOdd()) {
        fe_negate(p.y, p.y);
    }

    p.infinity = false;
    return true;
}

// Get x-coordinate as 32 bytes
__device__ __host__ inline void pubkey_get_xonly(uint8_t* out, const AffinePoint& p) {
    p.x.GetBytes(out);
}

// Check if y-coordinate is even (for BIP340)
__device__ __host__ inline bool pubkey_has_even_y(const AffinePoint& p) {
    return !p.y.IsOdd();
}

} // namespace secp256k1
} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SECP256K1_ECMULT_CUH
