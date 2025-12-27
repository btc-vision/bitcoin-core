// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SECP256K1_GROUP_CUH
#define BITCOIN_GPU_KERNEL_GPU_SECP256K1_GROUP_CUH

#include "gpu_secp256k1_field.cuh"
#include <cstdint>

namespace gpu {
namespace secp256k1 {

// ============================================================================
// secp256k1 Elliptic Curve Point
// y^2 = x^3 + 7 (mod p)
// ============================================================================

// Affine point (x, y), with infinity flag
struct AffinePoint {
    FieldElement x;
    FieldElement y;
    bool infinity;

    __device__ __host__ AffinePoint() : infinity(true) {}

    __device__ __host__ void SetInfinity() {
        infinity = true;
        x.SetZero();
        y.SetZero();
    }

    __device__ __host__ bool IsInfinity() const {
        return infinity;
    }

    __device__ __host__ bool IsOnCurve() const {
        if (infinity) return true;

        // Check y^2 = x^3 + 7
        FieldElement y2, x3, x3_plus_7, seven;

        fe_sqr(y2, y);

        fe_sqr(x3, x);
        fe_mul(x3, x3, x);

        seven.SetZero();
        seven.d[0] = 7;

        fe_add(x3_plus_7, x3, seven);

        return y2.IsEqual(x3_plus_7);
    }

    __device__ __host__ void Negate() {
        if (!infinity) {
            fe_negate(y, y);
        }
    }
};

// Jacobian point (X, Y, Z) where (x, y) = (X/Z^2, Y/Z^3)
// Point at infinity: Z = 0
struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;

    __device__ __host__ JacobianPoint() {
        x.SetZero();
        y.SetOne(); // Convention: infinity is (1, 1, 0)
        z.SetZero();
    }

    __device__ __host__ void SetInfinity() {
        x.SetZero();
        y.SetOne();
        z.SetZero();
    }

    __device__ __host__ bool IsInfinity() const {
        return z.IsZero();
    }

    // Convert from affine
    __device__ __host__ void FromAffine(const AffinePoint& p) {
        if (p.infinity) {
            SetInfinity();
        } else {
            x = p.x;
            y = p.y;
            z.SetOne();
        }
    }

    // Convert to affine
    __device__ __host__ void ToAffine(AffinePoint& p) const {
        if (IsInfinity()) {
            p.SetInfinity();
            return;
        }

        p.infinity = false;

        // Compute z^(-1)
        FieldElement z_inv, z_inv2, z_inv3;
        fe_inv(z_inv, z);

        // x = X / Z^2
        fe_sqr(z_inv2, z_inv);
        fe_mul(p.x, x, z_inv2);

        // y = Y / Z^3
        fe_mul(z_inv3, z_inv2, z_inv);
        fe_mul(p.y, y, z_inv3);
    }

    // Negate: (X, Y, Z) -> (X, -Y, Z)
    __device__ __host__ void Negate() {
        if (!IsInfinity()) {
            fe_negate(y, y);
        }
    }
};

// ============================================================================
// Generator Point G
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
// ============================================================================

__device__ __host__ inline void GetGenerator(AffinePoint& g) {
    g.infinity = false;

    // Gx (little-endian limbs)
    g.x.d[0] = 0x16F81798;
    g.x.d[1] = 0x59F2815B;
    g.x.d[2] = 0x2DCE28D9;
    g.x.d[3] = 0x029BFCDB;
    g.x.d[4] = 0xCE870B07;
    g.x.d[5] = 0x55A06295;
    g.x.d[6] = 0xF9DCBBAC;
    g.x.d[7] = 0x79BE667E;

    // Gy (little-endian limbs)
    g.y.d[0] = 0xFB10D4B8;
    g.y.d[1] = 0x9C47D08F;
    g.y.d[2] = 0xA6855419;
    g.y.d[3] = 0xFD17B448;
    g.y.d[4] = 0x0E1108A8;
    g.y.d[5] = 0x5DA4FBFC;
    g.y.d[6] = 0x26A3C465;
    g.y.d[7] = 0x483ADA77;
}

// ============================================================================
// Group Operations
// ============================================================================

// Point doubling: r = 2 * p (Jacobian coordinates)
// Uses the formula from "Guide to ECC", algorithm 3.21
// Cost: 1M + 5S + 1*a + 7add + 2*2 + 1*3 + 1*8
// For secp256k1, a = 0, so we save a multiplication
// Note: Handles aliasing (r == p) correctly
__device__ __host__ inline void point_double(JacobianPoint& r, const JacobianPoint& p) {
    if (p.IsInfinity()) {
        r.SetInfinity();
        return;
    }

    FieldElement t1, t2, t3, t4, t5;

    // Save p.y and p.z early since we'll overwrite r.y before using them for Z3
    // (needed when r aliases p)
    FieldElement py = p.y;
    FieldElement pz = p.z;

    // t1 = Y^2
    fe_sqr(t1, py);

    // t2 = 4 * X * Y^2
    fe_mul(t2, p.x, t1);
    fe_add(t2, t2, t2);  // 2 * X * Y^2
    fe_add(t2, t2, t2);  // 4 * X * Y^2

    // t3 = 8 * Y^4
    fe_sqr(t3, t1);      // Y^4
    fe_add(t3, t3, t3);  // 2 * Y^4
    fe_add(t3, t3, t3);  // 4 * Y^4
    fe_add(t3, t3, t3);  // 8 * Y^4

    // t4 = 3 * X^2 (since a = 0 for secp256k1)
    fe_sqr(t4, p.x);     // X^2
    fe_add(t5, t4, t4);  // 2 * X^2
    fe_add(t4, t5, t4);  // 3 * X^2

    // X3 = t4^2 - 2 * t2
    fe_sqr(r.x, t4);     // t4^2
    fe_sub(r.x, r.x, t2);
    fe_sub(r.x, r.x, t2);

    // Y3 = t4 * (t2 - X3) - t3
    fe_sub(t5, t2, r.x);
    fe_mul(r.y, t4, t5);
    fe_sub(r.y, r.y, t3);

    // Z3 = 2 * Y * Z (use saved py, pz)
    fe_mul(r.z, py, pz);
    fe_add(r.z, r.z, r.z);
}

// Point addition: r = p + q (Jacobian + Affine -> Jacobian)
// This is faster when one input is in affine form
// Cost: 7M + 4S + 9add + 3*2 + 1*4
__device__ __host__ inline void point_add_mixed(JacobianPoint& r, const JacobianPoint& p, const AffinePoint& q) {
    if (q.infinity) {
        r = p;
        return;
    }

    if (p.IsInfinity()) {
        r.x = q.x;
        r.y = q.y;
        r.z.SetOne();
        return;
    }

    FieldElement z2, u2, s2, h, hh, i, j, rr, v, t1;

    // z2 = Z1^2
    fe_sqr(z2, p.z);

    // u2 = X2 * Z1^2
    fe_mul(u2, q.x, z2);

    // s2 = Y2 * Z1^3
    fe_mul(s2, q.y, z2);
    fe_mul(s2, s2, p.z);

    // h = U2 - X1
    fe_sub(h, u2, p.x);

    // Check if points are the same (h == 0)
    if (h.IsZero()) {
        // Check if s2 == y1
        if (s2.IsEqual(p.y)) {
            // Points are the same, use doubling
            point_double(r, p);
            return;
        } else {
            // Points are negatives, result is infinity
            r.SetInfinity();
            return;
        }
    }

    // i = (2*h)^2
    fe_add(hh, h, h);
    fe_sqr(i, hh);

    // j = h * i
    fe_mul(j, h, i);

    // rr = 2 * (S2 - Y1)
    fe_sub(rr, s2, p.y);
    fe_add(rr, rr, rr);

    // v = X1 * i
    fe_mul(v, p.x, i);

    // X3 = rr^2 - j - 2*v
    fe_sqr(r.x, rr);
    fe_sub(r.x, r.x, j);
    fe_sub(r.x, r.x, v);
    fe_sub(r.x, r.x, v);

    // Y3 = rr * (v - X3) - 2 * Y1 * j
    fe_sub(t1, v, r.x);
    fe_mul(r.y, rr, t1);
    fe_mul(t1, p.y, j);
    fe_add(t1, t1, t1);
    fe_sub(r.y, r.y, t1);

    // Z3 = 2 * Z1 * h
    fe_mul(r.z, p.z, h);
    fe_add(r.z, r.z, r.z);
}

// Point addition: r = p + q (both in Jacobian coordinates)
// Cost: 11M + 5S + 9add + 4*2
__device__ __host__ inline void point_add(JacobianPoint& r, const JacobianPoint& p, const JacobianPoint& q) {
    if (p.IsInfinity()) {
        r = q;
        return;
    }
    if (q.IsInfinity()) {
        r = p;
        return;
    }

    FieldElement z1z1, z2z2, u1, u2, s1, s2, h, i, j, rr, v, t1;

    // Z1Z1 = Z1^2
    fe_sqr(z1z1, p.z);

    // Z2Z2 = Z2^2
    fe_sqr(z2z2, q.z);

    // U1 = X1 * Z2Z2
    fe_mul(u1, p.x, z2z2);

    // U2 = X2 * Z1Z1
    fe_mul(u2, q.x, z1z1);

    // S1 = Y1 * Z2 * Z2Z2
    fe_mul(s1, p.y, q.z);
    fe_mul(s1, s1, z2z2);

    // S2 = Y2 * Z1 * Z1Z1
    fe_mul(s2, q.y, p.z);
    fe_mul(s2, s2, z1z1);

    // H = U2 - U1
    fe_sub(h, u2, u1);

    // Check if points are the same or negatives
    if (h.IsZero()) {
        fe_sub(t1, s2, s1);
        if (t1.IsZero()) {
            // Points are the same
            point_double(r, p);
            return;
        } else {
            // Points are negatives
            r.SetInfinity();
            return;
        }
    }

    // I = (2*H)^2
    fe_add(i, h, h);
    fe_sqr(i, i);

    // J = H * I
    fe_mul(j, h, i);

    // rr = 2 * (S2 - S1)
    fe_sub(rr, s2, s1);
    fe_add(rr, rr, rr);

    // V = U1 * I
    fe_mul(v, u1, i);

    // X3 = rr^2 - J - 2*V
    fe_sqr(r.x, rr);
    fe_sub(r.x, r.x, j);
    fe_sub(r.x, r.x, v);
    fe_sub(r.x, r.x, v);

    // Y3 = rr * (V - X3) - 2 * S1 * J
    fe_sub(t1, v, r.x);
    fe_mul(r.y, rr, t1);
    fe_mul(t1, s1, j);
    fe_add(t1, t1, t1);
    fe_sub(r.y, r.y, t1);

    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    fe_add(r.z, p.z, q.z);
    fe_sqr(r.z, r.z);
    fe_sub(r.z, r.z, z1z1);
    fe_sub(r.z, r.z, z2z2);
    fe_mul(r.z, r.z, h);
}

// Point subtraction: r = p - q
__device__ __host__ inline void point_sub(JacobianPoint& r, const JacobianPoint& p, const JacobianPoint& q) {
    JacobianPoint neg_q = q;
    neg_q.Negate();
    point_add(r, p, neg_q);
}

// Check if two points are equal (both in Jacobian coordinates)
__device__ __host__ inline bool point_equal(const JacobianPoint& p, const JacobianPoint& q) {
    if (p.IsInfinity() && q.IsInfinity()) return true;
    if (p.IsInfinity() || q.IsInfinity()) return false;

    // Compare (X1/Z1^2, Y1/Z1^3) == (X2/Z2^2, Y2/Z2^3)
    // Without division: X1*Z2^2 == X2*Z1^2 && Y1*Z2^3 == Y2*Z1^3

    FieldElement z1z1, z2z2, u1, u2, s1, s2;

    fe_sqr(z1z1, p.z);
    fe_sqr(z2z2, q.z);

    fe_mul(u1, p.x, z2z2);
    fe_mul(u2, q.x, z1z1);

    if (!u1.IsEqual(u2)) return false;

    fe_mul(s1, p.y, q.z);
    fe_mul(s1, s1, z2z2);

    fe_mul(s2, q.y, p.z);
    fe_mul(s2, s2, z1z1);

    return s1.IsEqual(s2);
}

// Normalize Z coordinate to 1 (convert to quasi-affine)
__device__ __host__ inline void point_normalize(JacobianPoint& p) {
    if (p.IsInfinity()) return;

    FieldElement z_inv, z_inv2, z_inv3;

    fe_inv(z_inv, p.z);
    fe_sqr(z_inv2, z_inv);
    fe_mul(z_inv3, z_inv2, z_inv);

    fe_mul(p.x, p.x, z_inv2);
    fe_mul(p.y, p.y, z_inv3);
    p.z.SetOne();
}

// Check if point is on the curve (Jacobian coordinates)
__device__ __host__ inline bool point_is_valid(const JacobianPoint& p) {
    if (p.IsInfinity()) return true;

    // Check Y^2 = X^3 + 7*Z^6
    FieldElement y2, x3, z2, z4, z6, rhs, seven;

    fe_sqr(y2, p.y);

    fe_sqr(x3, p.x);
    fe_mul(x3, x3, p.x);

    fe_sqr(z2, p.z);
    fe_sqr(z4, z2);
    fe_mul(z6, z4, z2);

    seven.SetZero();
    seven.d[0] = 7;
    fe_mul(rhs, seven, z6);
    fe_add(rhs, x3, rhs);

    return y2.IsEqual(rhs);
}

// Get the x-coordinate of a point (normalized)
__device__ __host__ inline void point_get_x(FieldElement& x, const JacobianPoint& p) {
    if (p.IsInfinity()) {
        x.SetZero();
        return;
    }

    FieldElement z_inv, z_inv2;
    fe_inv(z_inv, p.z);
    fe_sqr(z_inv2, z_inv);
    fe_mul(x, p.x, z_inv2);
}

// Get the y-coordinate of a point (normalized)
__device__ __host__ inline void point_get_y(FieldElement& y, const JacobianPoint& p) {
    if (p.IsInfinity()) {
        y.SetZero();
        return;
    }

    FieldElement z_inv, z_inv2, z_inv3;
    fe_inv(z_inv, p.z);
    fe_sqr(z_inv2, z_inv);
    fe_mul(z_inv3, z_inv2, z_inv);
    fe_mul(y, p.y, z_inv3);
}

// Conditional negate: if flag, negate the point
__device__ __host__ inline void point_cnegate(JacobianPoint& p, bool flag) {
    if (flag && !p.IsInfinity()) {
        fe_negate(p.y, p.y);
    }
}

} // namespace secp256k1
} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SECP256K1_GROUP_CUH
