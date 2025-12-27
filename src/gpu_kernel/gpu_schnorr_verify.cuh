// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SCHNORR_VERIFY_CUH
#define BITCOIN_GPU_KERNEL_GPU_SCHNORR_VERIFY_CUH

#include "gpu_secp256k1_ecmult.cuh"
#include "gpu_hash.cuh"
#include <cstdint>

namespace gpu {
namespace secp256k1 {

// ============================================================================
// BIP340 Schnorr Signature Verification
// ============================================================================

// Tagged hash: SHA256(SHA256(tag) || SHA256(tag) || data)
// This is used in BIP340 for domain separation
// Note: Uses fixed buffer, max data size is 256 bytes
__device__ __host__ inline void tagged_hash(
    uint8_t* out,
    const char* tag,
    const uint8_t* data,
    uint32_t data_len)
{
    // Limit data_len to avoid buffer overflow
    if (data_len > 256) data_len = 256;

    // Compute SHA256(tag)
    uint8_t tag_hash[32];
    uint32_t tag_len = 0;
    while (tag[tag_len]) tag_len++;
    sha256((const uint8_t*)tag, tag_len, tag_hash);

    // Compute SHA256(tag_hash || tag_hash || data)
    // Use fixed-size buffer to avoid dynamic allocation (required for GPU)
    uint8_t buffer[64 + 256];
    for (int i = 0; i < 32; i++) buffer[i] = tag_hash[i];
    for (int i = 0; i < 32; i++) buffer[32 + i] = tag_hash[i];
    for (uint32_t i = 0; i < data_len; i++) buffer[64 + i] = data[i];

    sha256(buffer, 64 + data_len, out);
}

// Precomputed BIP340 tag hashes (for efficiency)
// SHA256("BIP0340/challenge")
#ifdef __CUDA_ARCH__
__device__ __constant__ const uint8_t BIP340_CHALLENGE_TAG_HASH[32] = {
#else
static constexpr uint8_t BIP340_CHALLENGE_TAG_HASH[32] = {
#endif
    0x7b, 0xb5, 0x2d, 0x7a, 0x9f, 0xef, 0x58, 0x32,
    0x3e, 0xb1, 0xbf, 0x7a, 0x40, 0x7d, 0xb3, 0x82,
    0xd2, 0xf3, 0xf2, 0xd8, 0x1b, 0xb1, 0x22, 0x4f,
    0x49, 0xfe, 0x51, 0x8f, 0x6d, 0x48, 0xd3, 0x7c
};

// BIP340 challenge hash: SHA256(tag || tag || r || P || m)
// where tag = SHA256("BIP0340/challenge")
__device__ __host__ inline void bip340_challenge_hash(
    uint8_t* e,
    const uint8_t* r_bytes,  // 32 bytes
    const uint8_t* p_bytes,  // 32 bytes (x-only pubkey)
    const uint8_t* msg,      // 32 bytes message
    uint32_t msg_len = 32)
{
    // buffer = tag_hash || tag_hash || r || P || m
    uint8_t buffer[64 + 32 + 32 + 32];

    // Copy tag hash twice
    for (int i = 0; i < 32; i++) {
        buffer[i] = BIP340_CHALLENGE_TAG_HASH[i];
        buffer[32 + i] = BIP340_CHALLENGE_TAG_HASH[i];
    }

    // Copy r, P, m
    for (int i = 0; i < 32; i++) buffer[64 + i] = r_bytes[i];
    for (int i = 0; i < 32; i++) buffer[96 + i] = p_bytes[i];
    for (uint32_t i = 0; i < msg_len; i++) buffer[128 + i] = msg[i];

    sha256(buffer, 128 + msg_len, e);
}

// ============================================================================
// Schnorr Signature Verification (BIP340)
// ============================================================================

// Parse a BIP340 Schnorr signature (64 bytes: r || s)
__device__ __host__ inline bool schnorr_parse_sig(
    FieldElement& r,
    Scalar& s,
    const uint8_t* sig)
{
    // First 32 bytes: r (field element, x-coordinate)
    r.SetBytes(sig);

    // Check r is valid field element (< p)
    if (fe_cmp_p(r) >= 0) return false;

    // Last 32 bytes: s (scalar)
    s.SetBytes(sig + 32);

    // Check s is valid scalar (< n)
    if (scalar_cmp_n(s) >= 0) return false;

    return true;
}

// BIP340 Schnorr verification core
// Inputs:
//   - sig: 64-byte signature (r || s)
//   - msg: 32-byte message hash
//   - pubkey: 32-byte x-only public key
// Returns true if signature is valid
__device__ __host__ inline bool schnorr_verify_core(
    const uint8_t* sig,
    const uint8_t* msg,
    const uint8_t* pubkey_bytes)
{
    // Parse public key (x-only)
    AffinePoint P;
    if (!pubkey_parse_xonly(P, pubkey_bytes)) {
        return false;
    }

    // Parse signature
    FieldElement r;
    Scalar s;
    if (!schnorr_parse_sig(r, s, sig)) {
        return false;
    }

    // Compute challenge e = int(hash_BIP0340/challenge(r || P || m)) mod n
    uint8_t e_bytes[32];
    uint8_t r_bytes[32];
    r.GetBytes(r_bytes);
    bip340_challenge_hash(e_bytes, r_bytes, pubkey_bytes, msg);

    Scalar e;
    e.SetBytes(e_bytes);
    scalar_reduce(e);

    // Compute R = s*G - e*P
    AffinePoint g_affine;
    GetGenerator(g_affine);

    JacobianPoint g_jac;
    g_jac.FromAffine(g_affine);

    JacobianPoint p_jac;
    p_jac.FromAffine(P);

    // Compute s*G
    JacobianPoint sG;
    ecmult_simple(sG, g_jac, s);

    // Compute e*P
    JacobianPoint eP;
    ecmult_simple(eP, p_jac, e);

    // Compute R = s*G - e*P
    JacobianPoint neg_eP = eP;
    neg_eP.Negate();

    JacobianPoint R;
    point_add(R, sG, neg_eP);

    // R must not be infinity
    if (R.IsInfinity()) return false;

    // Get R in affine coordinates
    AffinePoint R_affine;
    R.ToAffine(R_affine);

    // R.y must be even (lift_x returns the even y)
    if (R_affine.y.IsOdd()) return false;

    // R.x must equal r
    return R_affine.x.IsEqual(r);
}

// Full BIP340 Schnorr verification
// sig: 64-byte signature
// msg: message (variable length, typically 32 bytes)
// pubkey: 32-byte x-only public key
__device__ __host__ inline bool schnorr_verify(
    const uint8_t* sig,
    const uint8_t* msg,
    uint32_t msg_len,
    const uint8_t* pubkey)
{
    // For BIP341/342 (Taproot), message is 32 bytes
    if (msg_len != 32) {
        // For non-32-byte messages, need different handling
        // For now, only support 32-byte messages
        return false;
    }

    return schnorr_verify_core(sig, msg, pubkey);
}

// Schnorr verification with precomputed generator table (FAST)
__device__ __host__ inline bool schnorr_verify_fast(
    const uint8_t* sig,
    const uint8_t* msg,
    uint32_t msg_len,
    const uint8_t* pubkey,
    const GeneratorTable& gen_table)
{
    if (msg_len != 32) return false;

    // Parse signature (r, s)
    FieldElement r;
    r.SetBytes(sig);  // First 32 bytes

    Scalar s;
    s.SetBytes(sig + 32);  // Second 32 bytes

    // s must be < n
    if (scalar_cmp_n(s) >= 0) return false;

    // Parse public key (lift_x)
    AffinePoint P;
    if (!pubkey_parse_xonly(P, pubkey)) return false;

    // Compute e = hash(r || P || m)
    uint8_t e_bytes[32];
    bip340_challenge_hash(e_bytes, sig, pubkey, msg);

    Scalar e;
    e.SetBytes(e_bytes);
    scalar_reduce(e);

    // Compute R = s*G - e*P using precomputed table for G
    JacobianPoint p_jac;
    p_jac.FromAffine(P);

    // s*G using precomputed table (FAST!)
    JacobianPoint sG;
    ecmult_gen(sG, gen_table, s);

    // e*P
    JacobianPoint eP;
    ecmult_simple(eP, p_jac, e);

    // R = s*G - e*P
    JacobianPoint neg_eP = eP;
    neg_eP.Negate();

    JacobianPoint R;
    point_add(R, sG, neg_eP);

    if (R.IsInfinity()) return false;

    AffinePoint R_affine;
    R.ToAffine(R_affine);

    if (R_affine.y.IsOdd()) return false;

    return R_affine.x.IsEqual(r);
}

// ============================================================================
// Batch Schnorr Verification
// ============================================================================

struct SchnorrVerifyJob {
    const uint8_t* sig;        // 64 bytes
    const uint8_t* msg;        // 32 bytes
    const uint8_t* pubkey;     // 32 bytes (x-only)
    bool result;
    bool processed;

    __device__ __host__ SchnorrVerifyJob() :
        sig(nullptr), msg(nullptr), pubkey(nullptr),
        result(false), processed(false) {}
};

// Batch Schnorr verification
// Verifies multiple signatures in parallel
__device__ __host__ inline int schnorr_verify_batch(
    SchnorrVerifyJob* jobs,
    int count)
{
    int valid_count = 0;

    for (int i = 0; i < count; i++) {
        SchnorrVerifyJob& job = jobs[i];

        job.result = schnorr_verify_core(
            job.sig,
            job.msg,
            job.pubkey
        );
        job.processed = true;

        if (job.result) valid_count++;
    }

    return valid_count;
}

// ============================================================================
// Taproot-specific helpers
// ============================================================================

// Compute the Taproot tweak for a public key
// tweak = tagged_hash("TapTweak", P || merkle_root)
// Returns the tweaked public key
__device__ __host__ inline bool taproot_tweak_pubkey(
    AffinePoint& out,
    const AffinePoint& internal_key,
    const uint8_t* merkle_root)  // 32 bytes, can be nullptr for key-path-only
{
    // Get x-coordinate of internal key
    uint8_t pk_bytes[32];
    internal_key.x.GetBytes(pk_bytes);

    // Compute tweak = tagged_hash("TapTweak", pk || merkle_root)
    uint8_t tweak_data[64];
    for (int i = 0; i < 32; i++) tweak_data[i] = pk_bytes[i];
    if (merkle_root) {
        for (int i = 0; i < 32; i++) tweak_data[32 + i] = merkle_root[i];
    }

    uint8_t tweak[32];
    tagged_hash(tweak, "TapTweak", tweak_data, merkle_root ? 64 : 32);

    // Convert tweak to scalar
    Scalar t;
    t.SetBytes(tweak);

    // Check tweak is valid
    if (scalar_cmp_n(t) >= 0) return false;

    // Compute P' = P + t*G
    AffinePoint g;
    GetGenerator(g);

    JacobianPoint g_jac;
    g_jac.FromAffine(g);

    JacobianPoint tG;
    ecmult_simple(tG, g_jac, t);

    JacobianPoint p_jac;
    p_jac.FromAffine(internal_key);

    JacobianPoint result;
    point_add(result, p_jac, tG);

    if (result.IsInfinity()) return false;

    result.ToAffine(out);
    return true;
}

// ============================================================================
// Taproot Script-Path Merkle Proof Verification (BIP341)
// ============================================================================

// Taproot control block format:
// - 1 byte: leaf_version (top 7 bits) + output_parity (low bit)
// - 32 bytes: internal pubkey (x-only)
// - 32*k bytes: merkle path (0 <= k <= 128)

// Compute TapLeaf hash: tagged_hash("TapLeaf", leaf_version || compact_size(script) || script)
// Uses incremental hashing to avoid large stack allocation for big tapscripts
__device__ __host__ inline void compute_tapleaf_hash(
    uint8_t* out,                    // 32 bytes output
    uint8_t leaf_version,
    const uint8_t* script,
    uint32_t script_len)
{
    // Compute SHA256("TapLeaf") at runtime - no precomputed constants
    uint8_t tapleaf_tag[32];
    sha256((const uint8_t*)"TapLeaf", 7, tapleaf_tag);

    // Build prefix: leaf_version || compact_size(script_len)
    uint8_t prefix[6];
    uint32_t prefix_len = 0;
    prefix[prefix_len++] = leaf_version;

    // Compact size encoding
    if (script_len < 253) {
        prefix[prefix_len++] = (uint8_t)script_len;
    } else if (script_len <= 0xFFFF) {
        prefix[prefix_len++] = 253;
        prefix[prefix_len++] = script_len & 0xFF;
        prefix[prefix_len++] = (script_len >> 8) & 0xFF;
    } else {
        prefix[prefix_len++] = 254;
        prefix[prefix_len++] = script_len & 0xFF;
        prefix[prefix_len++] = (script_len >> 8) & 0xFF;
        prefix[prefix_len++] = (script_len >> 16) & 0xFF;
        prefix[prefix_len++] = (script_len >> 24) & 0xFF;
    }

    // Initialize SHA256 with standard IV
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Total length = 64 (two tag hashes) + prefix_len + script_len
    uint64_t total_len = 64 + prefix_len + script_len;

    // First block: SHA256("TapLeaf") || SHA256("TapLeaf") (64 bytes exactly)
    uint8_t block[64];
    for (int i = 0; i < 32; i++) block[i] = tapleaf_tag[i];
    for (int i = 0; i < 32; i++) block[32 + i] = tapleaf_tag[i];
    sha256_transform(state, block);

    // Process remaining data in 64-byte blocks
    uint32_t block_pos = 0;

    // Add prefix
    for (uint32_t i = 0; i < prefix_len; i++) {
        block[block_pos++] = prefix[i];
        if (block_pos == 64) {
            sha256_transform(state, block);
            block_pos = 0;
        }
    }

    // Add script
    for (uint32_t i = 0; i < script_len; i++) {
        block[block_pos++] = script[i];
        if (block_pos == 64) {
            sha256_transform(state, block);
            block_pos = 0;
        }
    }

    // Pad and finalize
    block[block_pos++] = 0x80;
    if (block_pos > 56) {
        while (block_pos < 64) block[block_pos++] = 0;
        sha256_transform(state, block);
        block_pos = 0;
    }
    while (block_pos < 56) block[block_pos++] = 0;

    // Append length in bits (big-endian)
    uint64_t bit_len = total_len * 8;
    block[56] = (bit_len >> 56) & 0xFF;
    block[57] = (bit_len >> 48) & 0xFF;
    block[58] = (bit_len >> 40) & 0xFF;
    block[59] = (bit_len >> 32) & 0xFF;
    block[60] = (bit_len >> 24) & 0xFF;
    block[61] = (bit_len >> 16) & 0xFF;
    block[62] = (bit_len >> 8) & 0xFF;
    block[63] = bit_len & 0xFF;
    sha256_transform(state, block);

    // Output hash (big-endian)
    for (int i = 0; i < 8; i++) {
        out[i * 4 + 0] = (state[i] >> 24) & 0xFF;
        out[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        out[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        out[i * 4 + 3] = state[i] & 0xFF;
    }
}

// Compute TapBranch hash: tagged_hash("TapBranch", sorted(left, right))
// The two 32-byte hashes are sorted lexicographically before hashing
__device__ __host__ inline void compute_tapbranch_hash(
    uint8_t* out,                    // 32 bytes output
    const uint8_t* hash_a,           // 32 bytes
    const uint8_t* hash_b)           // 32 bytes
{
    // Lexicographic comparison of hashes
    bool a_less_than_b = false;
    for (int i = 0; i < 32; i++) {
        if (hash_a[i] < hash_b[i]) {
            a_less_than_b = true;
            break;
        } else if (hash_a[i] > hash_b[i]) {
            a_less_than_b = false;
            break;
        }
    }

    // Concatenate in sorted order
    uint8_t data[64];
    if (a_less_than_b) {
        for (int i = 0; i < 32; i++) data[i] = hash_a[i];
        for (int i = 0; i < 32; i++) data[32 + i] = hash_b[i];
    } else {
        for (int i = 0; i < 32; i++) data[i] = hash_b[i];
        for (int i = 0; i < 32; i++) data[32 + i] = hash_a[i];
    }

    tagged_hash(out, "TapBranch", data, 64);
}

// Verify Taproot script-path merkle proof from control block
// Returns true if the script is committed to by the output pubkey
__device__ __host__ inline bool verify_taproot_script_path(
    const uint8_t* output_pubkey,    // 32 bytes, x-only output key from scriptPubKey
    const uint8_t* control_block,    // Control block from witness
    uint32_t control_block_len,
    const uint8_t* script,           // The tapscript being executed
    uint32_t script_len)
{
    // Minimum control block: 33 bytes (1 + 32)
    // Maximum: 33 + 128*32 = 4129 bytes (depth 128)
    if (control_block_len < 33) return false;
    if ((control_block_len - 33) % 32 != 0) return false;

    uint32_t path_len = (control_block_len - 33) / 32;
    if (path_len > 128) return false;  // Max tree depth

    // Parse control block
    uint8_t leaf_version = control_block[0] & 0xFE;  // Top 7 bits
    uint8_t output_parity = control_block[0] & 0x01;  // Low bit
    const uint8_t* internal_pubkey = control_block + 1;  // 32 bytes
    const uint8_t* path = control_block + 33;  // merkle path

    // Compute tapleaf hash
    uint8_t current_hash[32];
    compute_tapleaf_hash(current_hash, leaf_version, script, script_len);

    // Walk up the merkle tree
    for (uint32_t i = 0; i < path_len; i++) {
        const uint8_t* sibling = path + (i * 32);
        compute_tapbranch_hash(current_hash, current_hash, sibling);
    }

    // current_hash is now the merkle root
    // Compute the tweaked public key and verify it matches output_pubkey

    // Parse internal key
    AffinePoint internal;
    if (!pubkey_parse_xonly(internal, internal_pubkey)) return false;

    // Compute tweaked key P' = internal + tweak*G
    AffinePoint tweaked;
    if (!taproot_tweak_pubkey(tweaked, internal, current_hash)) return false;

    // Get x-coordinate and parity of tweaked key
    uint8_t tweaked_x[32];
    tweaked.x.GetBytes(tweaked_x);

    // Check x-coordinate matches output pubkey
    for (int i = 0; i < 32; i++) {
        if (tweaked_x[i] != output_pubkey[i]) return false;
    }

    // Check parity matches
    uint8_t tweaked_parity = tweaked.y.IsOdd() ? 1 : 0;
    if (tweaked_parity != output_parity) return false;

    return true;
}

// Get the internal pubkey from control block (for signature verification)
__device__ __host__ inline bool get_internal_pubkey_from_control_block(
    uint8_t* internal_pubkey_out,     // 32 bytes output
    uint8_t* leaf_version_out,
    const uint8_t* control_block,
    uint32_t control_block_len)
{
    if (control_block_len < 33) return false;

    *leaf_version_out = control_block[0] & 0xFE;
    for (int i = 0; i < 32; i++) {
        internal_pubkey_out[i] = control_block[1 + i];
    }
    return true;
}

// Check if a public key commitment matches a given output key and merkle root
__device__ __host__ inline bool taproot_verify_commitment(
    const uint8_t* output_pubkey,    // 32 bytes, x-only output key
    const uint8_t* internal_pubkey,  // 32 bytes, x-only internal key
    const uint8_t* merkle_root)      // 32 bytes, can be nullptr
{
    // Parse internal key
    AffinePoint internal;
    if (!pubkey_parse_xonly(internal, internal_pubkey)) {
        return false;
    }

    // Compute tweaked key
    AffinePoint tweaked;
    if (!taproot_tweak_pubkey(tweaked, internal, merkle_root)) {
        return false;
    }

    // Compare x-coordinate with output key
    uint8_t tweaked_x[32];
    tweaked.x.GetBytes(tweaked_x);

    for (int i = 0; i < 32; i++) {
        if (tweaked_x[i] != output_pubkey[i]) return false;
    }

    return true;
}

} // namespace secp256k1
} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SCHNORR_VERIFY_CUH
