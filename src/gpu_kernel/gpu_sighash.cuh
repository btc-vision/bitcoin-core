// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SIGHASH_CUH
#define BITCOIN_GPU_KERNEL_GPU_SIGHASH_CUH

#include "gpu_hash.cuh"
#include "gpu_schnorr_verify.cuh"
#include <cstdint>

namespace gpu {
namespace sighash {


// ============================================================================
// SIGHASH Types and Constants
// ============================================================================

// SIGHASH flags (same as Bitcoin Core)
constexpr uint8_t SIGHASH_ALL = 1;
constexpr uint8_t SIGHASH_NONE = 2;
constexpr uint8_t SIGHASH_SINGLE = 3;
constexpr uint8_t SIGHASH_ANYONECANPAY = 0x80;
constexpr uint8_t SIGHASH_DEFAULT = 0;  // Taproot only (equivalent to SIGHASH_ALL)

// Signature versions (for BIP143/341)
enum class SigVersion : uint8_t {
    BASE = 0,        // Legacy pre-segwit
    WITNESS_V0 = 1,  // BIP143 SegWit v0
    TAPROOT = 2,     // BIP341 Taproot key path
    TAPSCRIPT = 3    // BIP342 Tapscript
};

// ============================================================================
// Transaction Data Structures (GPU-optimized)
// ============================================================================

// Outpoint (txid + output index)
struct GPUOutPoint {
    uint8_t txid[32];
    uint32_t n;

    __device__ __host__ void Serialize(uint8_t* out) const {
        for (int i = 0; i < 32; i++) out[i] = txid[i];
        out[32] = n & 0xFF;
        out[33] = (n >> 8) & 0xFF;
        out[34] = (n >> 16) & 0xFF;
        out[35] = (n >> 24) & 0xFF;
    }
};

// Transaction input (simplified for GPU)
struct GPUTxIn {
    GPUOutPoint prevout;
    const uint8_t* scriptSig;
    uint32_t scriptSigLen;
    uint32_t nSequence;
};

// Transaction output (simplified for GPU)
struct GPUTxOut {
    int64_t nValue;
    const uint8_t* scriptPubKey;
    uint32_t scriptPubKeyLen;
};

// Transaction (simplified for GPU sighash computation)
struct GPUTransaction {
    int32_t nVersion;
    const GPUTxIn* vin;
    uint32_t vinCount;
    const GPUTxOut* vout;
    uint32_t voutCount;
    uint32_t nLockTime;
};

// Script execution context for sighash
struct GPUSigHashContext {
    const GPUTransaction* tx;
    uint32_t nIn;                    // Input being signed
    const uint8_t* scriptCode;       // Script being executed
    uint32_t scriptCodeLen;
    int64_t amount;                  // Amount of the UTXO being spent (for BIP143+)
    SigVersion sigversion;

    // Precomputed hashes (for efficiency)
    uint8_t hashPrevouts[32];
    uint8_t hashSequence[32];
    uint8_t hashOutputs[32];
    bool hashesComputed;

    // Taproot-specific
    uint8_t hashAmounts[32];
    uint8_t hashScriptPubKeys[32];
    const uint8_t* tapleafHash;       // 32 bytes if script path, nullptr for key path
    uint8_t keyVersion;               // 0 or 1
    uint32_t codeSeparatorPos;
};

// ============================================================================
// Helper Functions
// ============================================================================

// Write a 32-bit little-endian integer
__device__ __host__ inline void WriteLE32(uint8_t* out, uint32_t val) {
    out[0] = val & 0xFF;
    out[1] = (val >> 8) & 0xFF;
    out[2] = (val >> 16) & 0xFF;
    out[3] = (val >> 24) & 0xFF;
}

// Write a 64-bit little-endian integer
__device__ __host__ inline void WriteLE64(uint8_t* out, int64_t val) {
    uint64_t v = (uint64_t)val;
    out[0] = v & 0xFF;
    out[1] = (v >> 8) & 0xFF;
    out[2] = (v >> 16) & 0xFF;
    out[3] = (v >> 24) & 0xFF;
    out[4] = (v >> 32) & 0xFF;
    out[5] = (v >> 40) & 0xFF;
    out[6] = (v >> 48) & 0xFF;
    out[7] = (v >> 56) & 0xFF;
}

// Serialize a VarInt (compact size)
__device__ __host__ inline uint32_t WriteVarInt(uint8_t* out, uint64_t val) {
    if (val < 0xFD) {
        out[0] = (uint8_t)val;
        return 1;
    } else if (val <= 0xFFFF) {
        out[0] = 0xFD;
        out[1] = val & 0xFF;
        out[2] = (val >> 8) & 0xFF;
        return 3;
    } else if (val <= 0xFFFFFFFF) {
        out[0] = 0xFE;
        WriteLE32(out + 1, (uint32_t)val);
        return 5;
    } else {
        out[0] = 0xFF;
        WriteLE64(out + 1, val);
        return 9;
    }
}

// ============================================================================
// Precomputed Hash Computation (BIP143/341)
// ============================================================================

// Compute hashPrevouts = SHA256d(all prevouts)
__device__ __host__ inline void ComputeHashPrevouts(
    uint8_t* hashPrevouts,
    const GPUTransaction* tx)
{
    // Buffer for all prevouts: 36 bytes each (32 txid + 4 n)
    // Max reasonable inputs: ~100, so 3600 bytes
    uint8_t buffer[4096];
    uint32_t pos = 0;

    for (uint32_t i = 0; i < tx->vinCount && pos + 36 <= sizeof(buffer); i++) {
        for (int j = 0; j < 32; j++) {
            buffer[pos++] = tx->vin[i].prevout.txid[j];
        }
        WriteLE32(buffer + pos, tx->vin[i].prevout.n);
        pos += 4;
    }

    ::gpu::sha256d(buffer, pos, hashPrevouts);
}

// Compute hashSequence = SHA256d(all sequences)
__device__ __host__ inline void ComputeHashSequence(
    uint8_t* hashSequence,
    const GPUTransaction* tx)
{
    // Buffer for all sequences: 4 bytes each
    uint8_t buffer[1024];
    uint32_t pos = 0;

    for (uint32_t i = 0; i < tx->vinCount && pos + 4 <= sizeof(buffer); i++) {
        WriteLE32(buffer + pos, tx->vin[i].nSequence);
        pos += 4;
    }

    ::gpu::sha256d(buffer, pos, hashSequence);
}

// Compute hashOutputs = SHA256d(all outputs)
__device__ __host__ inline void ComputeHashOutputs(
    uint8_t* hashOutputs,
    const GPUTransaction* tx)
{
    // Buffer for all outputs
    uint8_t buffer[8192];
    uint32_t pos = 0;

    for (uint32_t i = 0; i < tx->voutCount; i++) {
        // Amount (8 bytes)
        WriteLE64(buffer + pos, tx->vout[i].nValue);
        pos += 8;

        // Script length (varint)
        pos += WriteVarInt(buffer + pos, tx->vout[i].scriptPubKeyLen);

        // Script
        if (pos + tx->vout[i].scriptPubKeyLen <= sizeof(buffer)) {
            for (uint32_t j = 0; j < tx->vout[i].scriptPubKeyLen; j++) {
                buffer[pos++] = tx->vout[i].scriptPubKey[j];
            }
        }
    }

    ::gpu::sha256d(buffer, pos, hashOutputs);
}

// Compute hashSingleOutput = SHA256d(output at nIn)
__device__ __host__ inline void ComputeHashSingleOutput(
    uint8_t* hashOutput,
    const GPUTransaction* tx,
    uint32_t nIn)
{
    if (nIn >= tx->voutCount) {
        // SIGHASH_SINGLE with no corresponding output - return 0x0100...00
        for (int i = 0; i < 32; i++) hashOutput[i] = 0;
        hashOutput[0] = 1;
        return;
    }

    uint8_t buffer[1024];
    uint32_t pos = 0;

    WriteLE64(buffer + pos, tx->vout[nIn].nValue);
    pos += 8;

    pos += WriteVarInt(buffer + pos, tx->vout[nIn].scriptPubKeyLen);

    for (uint32_t j = 0; j < tx->vout[nIn].scriptPubKeyLen && pos < sizeof(buffer); j++) {
        buffer[pos++] = tx->vout[nIn].scriptPubKey[j];
    }

    ::gpu::sha256d(buffer, pos, hashOutput);
}

// ============================================================================
// BIP341 Taproot-specific Precomputed Hashes
// ============================================================================

// Compute hashAmounts = SHA256(all input amounts)
__device__ __host__ inline void ComputeHashAmounts(
    uint8_t* hashAmounts,
    const int64_t* amounts,
    uint32_t count)
{
    uint8_t buffer[1024];
    uint32_t pos = 0;

    for (uint32_t i = 0; i < count && pos + 8 <= sizeof(buffer); i++) {
        WriteLE64(buffer + pos, amounts[i]);
        pos += 8;
    }

    ::gpu::sha256(buffer, pos, hashAmounts);
}

// Compute hashScriptPubKeys = SHA256(all input scriptPubKeys)
__device__ __host__ inline void ComputeHashScriptPubKeys(
    uint8_t* hashScriptPubKeys,
    const uint8_t** scriptPubKeys,
    const uint32_t* scriptPubKeyLens,
    uint32_t count)
{
    uint8_t buffer[8192];
    uint32_t pos = 0;

    for (uint32_t i = 0; i < count; i++) {
        pos += WriteVarInt(buffer + pos, scriptPubKeyLens[i]);
        for (uint32_t j = 0; j < scriptPubKeyLens[i] && pos < sizeof(buffer); j++) {
            buffer[pos++] = scriptPubKeys[i][j];
        }
    }

    ::gpu::sha256(buffer, pos, hashScriptPubKeys);
}

// ============================================================================
// Legacy Sighash (Pre-SegWit)
// ============================================================================

// Compute legacy sighash (pre-BIP143)
// This modifies a copy of the transaction based on sighash type
__device__ __host__ inline bool ComputeLegacySigHash(
    uint8_t* sighash,
    const GPUSigHashContext* ctx,
    uint8_t nHashType)
{
    const GPUTransaction* tx = ctx->tx;
    uint8_t baseType = nHashType & 0x1F;
    bool fAnyoneCanPay = (nHashType & SIGHASH_ANYONECANPAY) != 0;

    // Build the serialized transaction for signing
    uint8_t buffer[32768];
    uint32_t pos = 0;

    // Version
    WriteLE32(buffer + pos, tx->nVersion);
    pos += 4;

    // Input count
    if (fAnyoneCanPay) {
        buffer[pos++] = 1;  // Only signing one input
    } else {
        pos += WriteVarInt(buffer + pos, tx->vinCount);
    }

    // Inputs
    for (uint32_t i = 0; i < tx->vinCount; i++) {
        if (fAnyoneCanPay && i != ctx->nIn) continue;

        // Prevout
        for (int j = 0; j < 32; j++) buffer[pos++] = tx->vin[i].prevout.txid[j];
        WriteLE32(buffer + pos, tx->vin[i].prevout.n);
        pos += 4;

        // ScriptSig: use scriptCode for the input being signed, empty for others
        if (i == ctx->nIn) {
            pos += WriteVarInt(buffer + pos, ctx->scriptCodeLen);
            for (uint32_t j = 0; j < ctx->scriptCodeLen; j++) {
                buffer[pos++] = ctx->scriptCode[j];
            }
        } else {
            buffer[pos++] = 0;  // Empty script
        }

        // Sequence
        if (i == ctx->nIn) {
            WriteLE32(buffer + pos, tx->vin[i].nSequence);
        } else if (baseType == SIGHASH_NONE || baseType == SIGHASH_SINGLE) {
            // Other inputs get sequence 0 for NONE/SINGLE
            WriteLE32(buffer + pos, 0);
        } else {
            WriteLE32(buffer + pos, tx->vin[i].nSequence);
        }
        pos += 4;
    }

    // Output count
    if (baseType == SIGHASH_NONE) {
        buffer[pos++] = 0;  // No outputs
    } else if (baseType == SIGHASH_SINGLE) {
        if (ctx->nIn >= tx->voutCount) {
            // SIGHASH_SINGLE bug: return 1 followed by 31 zero bytes
            for (int i = 0; i < 32; i++) sighash[i] = 0;
            sighash[0] = 1;
            return true;
        }
        pos += WriteVarInt(buffer + pos, ctx->nIn + 1);
    } else {
        pos += WriteVarInt(buffer + pos, tx->voutCount);
    }

    // Outputs
    uint32_t nOutputs = (baseType == SIGHASH_NONE) ? 0 :
                        (baseType == SIGHASH_SINGLE) ? ctx->nIn + 1 : tx->voutCount;

    for (uint32_t i = 0; i < nOutputs; i++) {
        if (baseType == SIGHASH_SINGLE && i != ctx->nIn) {
            // Empty output: -1 satoshis, empty script
            WriteLE64(buffer + pos, -1);
            pos += 8;
            buffer[pos++] = 0;
        } else {
            WriteLE64(buffer + pos, tx->vout[i].nValue);
            pos += 8;
            pos += WriteVarInt(buffer + pos, tx->vout[i].scriptPubKeyLen);
            for (uint32_t j = 0; j < tx->vout[i].scriptPubKeyLen; j++) {
                buffer[pos++] = tx->vout[i].scriptPubKey[j];
            }
        }
    }

    // LockTime
    WriteLE32(buffer + pos, tx->nLockTime);
    pos += 4;

    // Hash type (4 bytes)
    WriteLE32(buffer + pos, nHashType);
    pos += 4;

    // Double SHA256
    ::gpu::sha256d(buffer, pos, sighash);
    return true;
}

// ============================================================================
// BIP143 SegWit v0 Sighash
// ============================================================================

// Compute BIP143 sighash (SegWit v0)
__device__ __host__ inline bool ComputeWitnessV0SigHash(
    uint8_t* sighash,
    GPUSigHashContext* ctx,
    uint8_t nHashType)
{
    const GPUTransaction* tx = ctx->tx;
    uint8_t baseType = nHashType & 0x1F;
    bool fAnyoneCanPay = (nHashType & SIGHASH_ANYONECANPAY) != 0;

    // Precompute hashes if not done
    if (!ctx->hashesComputed) {
        ComputeHashPrevouts(ctx->hashPrevouts, tx);
        ComputeHashSequence(ctx->hashSequence, tx);
        ComputeHashOutputs(ctx->hashOutputs, tx);
        ctx->hashesComputed = true;
    }

    // Build BIP143 serialization
    uint8_t buffer[1024];
    uint32_t pos = 0;

    // 1. nVersion (4 bytes)
    WriteLE32(buffer + pos, tx->nVersion);
    pos += 4;

    // 2. hashPrevouts (32 bytes)
    if (fAnyoneCanPay) {
        for (int i = 0; i < 32; i++) buffer[pos++] = 0;
    } else {
        for (int i = 0; i < 32; i++) buffer[pos++] = ctx->hashPrevouts[i];
    }

    // 3. hashSequence (32 bytes)
    if (fAnyoneCanPay || baseType == SIGHASH_SINGLE || baseType == SIGHASH_NONE) {
        for (int i = 0; i < 32; i++) buffer[pos++] = 0;
    } else {
        for (int i = 0; i < 32; i++) buffer[pos++] = ctx->hashSequence[i];
    }

    // 4. outpoint (36 bytes)
    for (int i = 0; i < 32; i++) buffer[pos++] = tx->vin[ctx->nIn].prevout.txid[i];
    WriteLE32(buffer + pos, tx->vin[ctx->nIn].prevout.n);
    pos += 4;

    // 5. scriptCode (var)
    pos += WriteVarInt(buffer + pos, ctx->scriptCodeLen);
    for (uint32_t i = 0; i < ctx->scriptCodeLen; i++) {
        buffer[pos++] = ctx->scriptCode[i];
    }

    // 6. value (8 bytes)
    WriteLE64(buffer + pos, ctx->amount);
    pos += 8;

    // 7. nSequence (4 bytes)
    WriteLE32(buffer + pos, tx->vin[ctx->nIn].nSequence);
    pos += 4;

    // 8. hashOutputs (32 bytes)
    if (baseType == SIGHASH_SINGLE) {
        if (ctx->nIn < tx->voutCount) {
            ComputeHashSingleOutput(buffer + pos, tx, ctx->nIn);
        } else {
            for (int i = 0; i < 32; i++) buffer[pos] = 0;
        }
        pos += 32;
    } else if (baseType == SIGHASH_NONE) {
        for (int i = 0; i < 32; i++) buffer[pos++] = 0;
    } else {
        for (int i = 0; i < 32; i++) buffer[pos++] = ctx->hashOutputs[i];
    }

    // 9. nLockTime (4 bytes)
    WriteLE32(buffer + pos, tx->nLockTime);
    pos += 4;

    // 10. nHashType (4 bytes)
    WriteLE32(buffer + pos, nHashType);
    pos += 4;

    // Double SHA256
    ::gpu::sha256d(buffer, pos, sighash);
    return true;
}

// ============================================================================
// BIP341 Taproot Sighash
// ============================================================================

// Precomputed BIP341 tag hash for "TapSighash"
#ifdef __CUDA_ARCH__
__device__ __constant__ const uint8_t TAPSIGHASH_TAG_HASH[32] = {
#else
static constexpr uint8_t TAPSIGHASH_TAG_HASH[32] = {
#endif
    0xf4, 0x0a, 0x48, 0xdf, 0x4b, 0x2a, 0x70, 0xc8,
    0xb4, 0x92, 0x4b, 0xf2, 0x65, 0x4d, 0x2e, 0x35,
    0x27, 0x32, 0x80, 0x01, 0x02, 0x8c, 0xd9, 0x35,
    0x8b, 0x2c, 0x51, 0x43, 0xf7, 0xf5, 0xdc, 0x28
};

// Compute BIP341 sighash (Taproot)
__device__ __host__ inline bool ComputeTaprootSigHash(
    uint8_t* sighash,
    GPUSigHashContext* ctx,
    uint8_t nHashType,
    const int64_t* allAmounts,           // All input amounts
    const uint8_t** allScriptPubKeys,    // All input scriptPubKeys
    const uint32_t* allScriptPubKeyLens) // Their lengths
{
    const GPUTransaction* tx = ctx->tx;
    uint8_t baseType = nHashType == 0 ? SIGHASH_ALL : (nHashType & 0x03);
    bool fAnyoneCanPay = (nHashType & SIGHASH_ANYONECANPAY) != 0;

    // Epoch
    uint8_t epoch = 0;

    // Build the sighash message
    uint8_t buffer[2048];
    uint32_t pos = 0;

    // Common prefix: epoch (1) + hashType (1) + version (4) + locktime (4)
    buffer[pos++] = epoch;
    buffer[pos++] = nHashType;
    WriteLE32(buffer + pos, tx->nVersion);
    pos += 4;
    WriteLE32(buffer + pos, tx->nLockTime);
    pos += 4;

    // If not ANYONECANPAY, include input-related hashes
    if (!fAnyoneCanPay) {
        // hashPrevouts (SHA256, not double)
        uint8_t hashPrevouts[32];
        {
            uint8_t tmp[4096] = {0};
            uint32_t tpos = 0;
            for (uint32_t i = 0; i < tx->vinCount; i++) {
                for (int j = 0; j < 32; j++) tmp[tpos++] = tx->vin[i].prevout.txid[j];
                WriteLE32(tmp + tpos, tx->vin[i].prevout.n);
                tpos += 4;
            }
            ::gpu::sha256(tmp, tpos, hashPrevouts);
        }
        for (int i = 0; i < 32; i++) buffer[pos++] = hashPrevouts[i];

        // hashAmounts (SHA256)
        uint8_t hashAmounts[32];
        ComputeHashAmounts(hashAmounts, allAmounts, tx->vinCount);
        for (int i = 0; i < 32; i++) buffer[pos++] = hashAmounts[i];

        // hashScriptPubKeys (SHA256)
        uint8_t hashScriptPubKeys[32];
        ComputeHashScriptPubKeys(hashScriptPubKeys, allScriptPubKeys, allScriptPubKeyLens, tx->vinCount);
        for (int i = 0; i < 32; i++) buffer[pos++] = hashScriptPubKeys[i];

        // hashSequences (SHA256, not double)
        uint8_t hashSequences[32];
        {
            uint8_t tmp[1024];
            uint32_t tpos = 0;
            for (uint32_t i = 0; i < tx->vinCount; i++) {
                WriteLE32(tmp + tpos, tx->vin[i].nSequence);
                tpos += 4;
            }
            ::gpu::sha256(tmp, tpos, hashSequences);
        }
        for (int i = 0; i < 32; i++) buffer[pos++] = hashSequences[i];
    }

    // If SIGHASH_ALL or SIGHASH_DEFAULT, include outputs hash
    if (baseType == SIGHASH_ALL) {
        // hashOutputs (SHA256, not double)
        uint8_t hashOutputs[32];
        {
            uint8_t tmp[8192];
            uint32_t tpos = 0;
            for (uint32_t i = 0; i < tx->voutCount; i++) {
                WriteLE64(tmp + tpos, tx->vout[i].nValue);
                tpos += 8;
                tpos += WriteVarInt(tmp + tpos, tx->vout[i].scriptPubKeyLen);
                for (uint32_t j = 0; j < tx->vout[i].scriptPubKeyLen; j++) {
                    tmp[tpos++] = tx->vout[i].scriptPubKey[j];
                }
            }
            ::gpu::sha256(tmp, tpos, hashOutputs);
        }
        for (int i = 0; i < 32; i++) buffer[pos++] = hashOutputs[i];
    }

    // Spend type (1 byte)
    // bit 0: ext_flag (1 if annex present - we don't support annex yet)
    // bit 1: tapleaf_hash present (script path spending)
    uint8_t spendType = 0;
    if (ctx->tapleafHash != nullptr) spendType |= 2;
    buffer[pos++] = spendType;

    // Input-specific data
    if (fAnyoneCanPay) {
        // Include full outpoint, amount, scriptPubKey, sequence
        for (int i = 0; i < 32; i++) buffer[pos++] = tx->vin[ctx->nIn].prevout.txid[i];
        WriteLE32(buffer + pos, tx->vin[ctx->nIn].prevout.n);
        pos += 4;
        WriteLE64(buffer + pos, allAmounts[ctx->nIn]);
        pos += 8;
        pos += WriteVarInt(buffer + pos, allScriptPubKeyLens[ctx->nIn]);
        for (uint32_t j = 0; j < allScriptPubKeyLens[ctx->nIn]; j++) {
            buffer[pos++] = allScriptPubKeys[ctx->nIn][j];
        }
        WriteLE32(buffer + pos, tx->vin[ctx->nIn].nSequence);
        pos += 4;
    } else {
        // Just the input index
        WriteLE32(buffer + pos, ctx->nIn);
        pos += 4;
    }

    // If SIGHASH_SINGLE, include the specific output
    if (baseType == SIGHASH_SINGLE) {
        if (ctx->nIn < tx->voutCount) {
            uint8_t outputHash[32];
            {
                uint8_t tmp[1024];
                uint32_t tpos = 0;
                WriteLE64(tmp + tpos, tx->vout[ctx->nIn].nValue);
                tpos += 8;
                tpos += WriteVarInt(tmp + tpos, tx->vout[ctx->nIn].scriptPubKeyLen);
                for (uint32_t j = 0; j < tx->vout[ctx->nIn].scriptPubKeyLen; j++) {
                    tmp[tpos++] = tx->vout[ctx->nIn].scriptPubKey[j];
                }
                ::gpu::sha256(tmp, tpos, outputHash);
            }
            for (int i = 0; i < 32; i++) buffer[pos++] = outputHash[i];
        } else {
            // No corresponding output
            return false;
        }
    }

    // Script path specific data
    if (ctx->tapleafHash != nullptr) {
        for (int i = 0; i < 32; i++) buffer[pos++] = ctx->tapleafHash[i];
        buffer[pos++] = ctx->keyVersion;
        WriteLE32(buffer + pos, ctx->codeSeparatorPos);
        pos += 4;
    }

    // Compute tagged hash: SHA256(tag || tag || message)
    uint8_t hashInput[64 + 2048];
    for (int i = 0; i < 32; i++) hashInput[i] = TAPSIGHASH_TAG_HASH[i];
    for (int i = 0; i < 32; i++) hashInput[32 + i] = TAPSIGHASH_TAG_HASH[i];
    for (uint32_t i = 0; i < pos; i++) hashInput[64 + i] = buffer[i];

    ::gpu::sha256(hashInput, 64 + pos, sighash);
    return true;
}

// ============================================================================
// Unified Sighash Interface
// ============================================================================

// Compute sighash based on signature version
__device__ __host__ inline bool ComputeSigHash(
    uint8_t* sighash,
    GPUSigHashContext* ctx,
    uint8_t nHashType,
    const int64_t* allAmounts = nullptr,
    const uint8_t** allScriptPubKeys = nullptr,
    const uint32_t* allScriptPubKeyLens = nullptr)
{
    switch (ctx->sigversion) {
        case SigVersion::BASE:
            return ComputeLegacySigHash(sighash, ctx, nHashType);

        case SigVersion::WITNESS_V0:
            return ComputeWitnessV0SigHash(sighash, ctx, nHashType);

        case SigVersion::TAPROOT:
        case SigVersion::TAPSCRIPT:
            if (allAmounts == nullptr || allScriptPubKeys == nullptr) {
                return false;
            }
            return ComputeTaprootSigHash(sighash, ctx, nHashType,
                                         allAmounts, allScriptPubKeys, allScriptPubKeyLens);

        default:
            return false;
    }
}

// ============================================================================
// Batch Sighash Computation (for parallel GPU execution)
// ============================================================================

struct SigHashJob {
    GPUSigHashContext ctx;
    uint8_t nHashType;
    uint8_t sighash[32];
    bool result;
    bool processed;

    // For Taproot
    const int64_t* allAmounts;
    const uint8_t** allScriptPubKeys;
    const uint32_t* allScriptPubKeyLens;
};

// Batch compute sighashes
__device__ __host__ inline int ComputeSigHashBatch(
    SigHashJob* jobs,
    int count)
{
    int successCount = 0;

    for (int i = 0; i < count; i++) {
        jobs[i].result = ComputeSigHash(
            jobs[i].sighash,
            &jobs[i].ctx,
            jobs[i].nHashType,
            jobs[i].allAmounts,
            jobs[i].allScriptPubKeys,
            jobs[i].allScriptPubKeyLens
        );
        jobs[i].processed = true;

        if (jobs[i].result) successCount++;
    }

    return successCount;
}

// ============================================================================
// Script Code Extraction (for P2PKH, P2WPKH, etc.)
// ============================================================================

// Extract script code for P2WPKH (creates OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG)
__device__ __host__ inline uint32_t BuildP2WPKHScriptCode(
    uint8_t* scriptCode,
    const uint8_t* pubkeyHash)  // 20 bytes
{
    scriptCode[0] = 0x76;  // OP_DUP
    scriptCode[1] = 0xA9;  // OP_HASH160
    scriptCode[2] = 0x14;  // Push 20 bytes
    for (int i = 0; i < 20; i++) {
        scriptCode[3 + i] = pubkeyHash[i];
    }
    scriptCode[23] = 0x88;  // OP_EQUALVERIFY
    scriptCode[24] = 0xAC;  // OP_CHECKSIG
    return 25;
}

// Extract script code for P2WSH (returns the witness script)
__device__ __host__ inline uint32_t BuildP2WSHScriptCode(
    uint8_t* scriptCode,
    const uint8_t* witnessScript,
    uint32_t witnessScriptLen)
{
    for (uint32_t i = 0; i < witnessScriptLen; i++) {
        scriptCode[i] = witnessScript[i];
    }
    return witnessScriptLen;
}

// Check if a scriptPubKey is P2WPKH (OP_0 <20 bytes>)
__device__ __host__ inline bool IsP2WPKH(const uint8_t* script, uint32_t len) {
    return len == 22 && script[0] == 0x00 && script[1] == 0x14;
}

// Check if a scriptPubKey is P2WSH (OP_0 <32 bytes>)
__device__ __host__ inline bool IsP2WSH(const uint8_t* script, uint32_t len) {
    return len == 34 && script[0] == 0x00 && script[1] == 0x20;
}

// Check if a scriptPubKey is P2TR (OP_1 <32 bytes>)
__device__ __host__ inline bool IsP2TR(const uint8_t* script, uint32_t len) {
    return len == 34 && script[0] == 0x51 && script[1] == 0x20;
}

} // namespace sighash
} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SIGHASH_CUH
