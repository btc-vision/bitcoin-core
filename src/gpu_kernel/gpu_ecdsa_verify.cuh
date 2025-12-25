// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_ECDSA_VERIFY_CUH
#define BITCOIN_GPU_KERNEL_GPU_ECDSA_VERIFY_CUH

#include "gpu_secp256k1_ecmult.cuh"
#include <cstdint>

namespace gpu {
namespace secp256k1 {

// ============================================================================
// DER Signature Parsing
// ============================================================================

// Parse a DER-encoded signature into (r, s) scalars
// DER format: 0x30 <total_len> 0x02 <r_len> <r_bytes> 0x02 <s_len> <s_bytes>
__device__ __host__ inline bool sig_parse_der(
    Scalar& r, Scalar& s,
    const uint8_t* sig, uint32_t sig_len)
{
    if (sig_len < 8 || sig_len > 73) return false;

    uint32_t pos = 0;

    // Check sequence tag
    if (sig[pos++] != 0x30) return false;

    // Get total length
    uint32_t total_len = sig[pos++];
    if (total_len + 2 != sig_len) return false;

    // Parse r
    if (sig[pos++] != 0x02) return false;
    uint32_t r_len = sig[pos++];
    if (r_len == 0 || r_len > 33) return false;
    if (pos + r_len > sig_len) return false;

    // Skip leading zero if present (used for positive sign)
    const uint8_t* r_data = sig + pos;
    if (r_len > 1 && r_data[0] == 0x00) {
        r_data++;
        r_len--;
    }
    pos += r_len + (r_data != sig + pos - r_len ? 1 : 0);

    // Convert r to scalar
    if (r_len > 32) return false;
    uint8_t r_bytes[32] = {0};
    for (uint32_t i = 0; i < r_len; i++) {
        r_bytes[32 - r_len + i] = r_data[i];
    }
    r.SetBytes(r_bytes);

    // Skip the leading zero we accounted for
    if (sig[pos - r_len - 1] == 0x00 && r_len < 32) {
        // Adjust position properly
    }

    // Recalculate position
    pos = 4 + sig[3];
    if (sig[4] == 0x00) pos++;

    // Parse s
    if (sig[pos++] != 0x02) return false;
    uint32_t s_len = sig[pos++];
    if (s_len == 0 || s_len > 33) return false;
    if (pos + s_len > sig_len) return false;

    // Skip leading zero if present
    const uint8_t* s_data = sig + pos;
    if (s_len > 1 && s_data[0] == 0x00) {
        s_data++;
        s_len--;
    }

    // Convert s to scalar
    if (s_len > 32) return false;
    uint8_t s_bytes[32] = {0};
    for (uint32_t i = 0; i < s_len; i++) {
        s_bytes[32 - s_len + i] = s_data[i];
    }
    s.SetBytes(s_bytes);

    return true;
}

// Simpler DER parsing with proper position tracking
__device__ __host__ inline bool sig_parse_der_simple(
    Scalar& r, Scalar& s,
    const uint8_t* sig, uint32_t sig_len)
{
    if (sig_len < 8 || sig_len > 73) return false;

    // 0x30 <len>
    if (sig[0] != 0x30) return false;
    uint32_t seq_len = sig[1];
    if (seq_len + 2 != sig_len) return false;

    uint32_t pos = 2;

    // Parse r: 0x02 <len> <data>
    if (sig[pos] != 0x02) return false;
    uint32_t r_len = sig[pos + 1];
    pos += 2;

    if (r_len == 0 || pos + r_len > sig_len) return false;

    // Parse r bytes (skip leading zero)
    const uint8_t* r_start = sig + pos;
    uint32_t r_actual_len = r_len;
    if (r_len > 0 && r_start[0] == 0x00) {
        r_start++;
        r_actual_len--;
    }
    if (r_actual_len > 32) return false;

    uint8_t r_bytes[32] = {0};
    for (uint32_t i = 0; i < r_actual_len; i++) {
        r_bytes[32 - r_actual_len + i] = r_start[i];
    }
    r.SetBytes(r_bytes);
    pos += r_len;

    // Parse s: 0x02 <len> <data>
    if (pos >= sig_len || sig[pos] != 0x02) return false;
    uint32_t s_len = sig[pos + 1];
    pos += 2;

    if (s_len == 0 || pos + s_len > sig_len) return false;

    // Parse s bytes (skip leading zero)
    const uint8_t* s_start = sig + pos;
    uint32_t s_actual_len = s_len;
    if (s_len > 0 && s_start[0] == 0x00) {
        s_start++;
        s_actual_len--;
    }
    if (s_actual_len > 32) return false;

    uint8_t s_bytes[32] = {0};
    for (uint32_t i = 0; i < s_actual_len; i++) {
        s_bytes[32 - s_actual_len + i] = s_start[i];
    }
    s.SetBytes(s_bytes);

    return true;
}

// ============================================================================
// ECDSA Verification
// ============================================================================

// ECDSA verification core algorithm
// Inputs:
//   - sighash: 32-byte message hash
//   - sig_r, sig_s: signature components
//   - pubkey: public key point
// Returns true if signature is valid
__device__ __host__ inline bool ecdsa_verify_core(
    const uint8_t* sighash,
    const Scalar& sig_r,
    const Scalar& sig_s,
    const AffinePoint& pubkey)
{
    // Check r and s are in range [1, n-1]
    if (sig_r.IsZero() || sig_s.IsZero()) return false;
    if (scalar_cmp_n(sig_r) >= 0) return false;
    if (scalar_cmp_n(sig_s) >= 0) return false;

    // Convert message hash to scalar
    Scalar e;
    e.SetBytes(sighash);
    scalar_reduce(e);

    // Compute w = s^(-1) mod n
    Scalar w;
    scalar_inv(w, sig_s);

    // Compute u1 = e * w mod n
    Scalar u1;
    scalar_mul(u1, e, w);

    // Compute u2 = r * w mod n
    Scalar u2;
    scalar_mul(u2, sig_r, w);

    // Compute R = u1 * G + u2 * Q
    // First, get generator point
    AffinePoint g_affine;
    GetGenerator(g_affine);

    JacobianPoint g_jac;
    g_jac.FromAffine(g_affine);

    JacobianPoint q_jac;
    q_jac.FromAffine(pubkey);

    // Compute u1 * G
    JacobianPoint r1;
    ecmult_simple(r1, g_jac, u1);

    // Compute u2 * Q
    JacobianPoint r2;
    ecmult_simple(r2, q_jac, u2);

    // R = r1 + r2
    JacobianPoint R;
    point_add(R, r1, r2);

    // Check if R is infinity
    if (R.IsInfinity()) return false;

    // Get R.x
    FieldElement rx;
    point_get_x(rx, R);

    // Convert rx to scalar (mod n)
    uint8_t rx_bytes[32];
    rx.GetBytes(rx_bytes);

    Scalar rx_scalar;
    rx_scalar.SetBytes(rx_bytes);
    scalar_reduce(rx_scalar);

    // Verify rx == r
    return rx_scalar.IsEqual(sig_r);
}

// Full ECDSA verification from raw inputs
// sig: DER-encoded signature
// sighash: 32-byte message hash
// pubkey: serialized public key (33 or 65 bytes)
__device__ __host__ inline bool ecdsa_verify(
    const uint8_t* sig, uint32_t sig_len,
    const uint8_t* sighash,
    const uint8_t* pubkey_data, uint32_t pubkey_len)
{
    // Parse signature
    Scalar r, s;
    if (!sig_parse_der_simple(r, s, sig, sig_len)) {
        return false;
    }

    // Parse public key
    AffinePoint pubkey;
    if (!pubkey_parse(pubkey, pubkey_data, pubkey_len)) {
        return false;
    }

    // Verify
    return ecdsa_verify_core(sighash, r, s, pubkey);
}

// ============================================================================
// Low-S Check (BIP 66 / BIP 146)
// ============================================================================

// Check if s is in the low half of the curve order
// For a valid signature, s should be <= n/2
__device__ __host__ inline bool sig_has_low_s(const Scalar& s) {
    return !s.IsHigh();
}

// Normalize s to low-S form if needed
// Returns the normalized s value
__device__ __host__ inline void sig_normalize_s(Scalar& s) {
    if (s.IsHigh()) {
        scalar_negate(s, s);
    }
}

// ============================================================================
// Batch Verification Structures
// ============================================================================

struct ECDSAVerifyJob {
    const uint8_t* sig;
    uint32_t sig_len;
    const uint8_t* sighash;
    const uint8_t* pubkey;
    uint32_t pubkey_len;
    bool result;
    bool processed;

    __device__ __host__ ECDSAVerifyJob() :
        sig(nullptr), sig_len(0), sighash(nullptr),
        pubkey(nullptr), pubkey_len(0), result(false), processed(false) {}
};

// Batch ECDSA verification
// Verifies multiple signatures in parallel
// Returns the number of valid signatures
__device__ __host__ inline int ecdsa_verify_batch(
    ECDSAVerifyJob* jobs,
    int count)
{
    int valid_count = 0;

    for (int i = 0; i < count; i++) {
        ECDSAVerifyJob& job = jobs[i];

        job.result = ecdsa_verify(
            job.sig, job.sig_len,
            job.sighash,
            job.pubkey, job.pubkey_len
        );
        job.processed = true;

        if (job.result) valid_count++;
    }

    return valid_count;
}

} // namespace secp256k1
} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_ECDSA_VERIFY_CUH
