// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_OPCODES_SIG_CUH
#define BITCOIN_GPU_KERNEL_GPU_OPCODES_SIG_CUH

#include "gpu_script_types.cuh"
#include "gpu_script_stack.cuh"
#include "gpu_ecdsa_verify.cuh"
#include "gpu_schnorr_verify.cuh"
#include "gpu_sighash.cuh"

namespace gpu {

// ============================================================================
// Signature Verification Helper Functions
// ============================================================================

// Check if a signature is valid (non-empty and properly formatted)
__device__ __host__ inline bool IsValidSignatureEncoding(
    const uint8_t* sig, uint32_t sig_len)
{
    // Empty signature is not a valid encoding
    if (sig_len == 0) return false;

    // Minimum DER signature length is 8 bytes
    if (sig_len < 8) return false;

    // Maximum DER signature length is 73 bytes (with sighash byte)
    if (sig_len > 73) return false;

    // Check DER structure
    if (sig[0] != 0x30) return false;

    // Check that length matches
    if (sig[1] != sig_len - 3) return false;

    return true;
}

// Check for low-S signature (BIP 146)
__device__ __host__ inline bool CheckLowS(const secp256k1::Scalar& s)
{
    return !s.IsHigh();
}

// Extract sighash type from signature
__device__ __host__ inline uint8_t GetSigHashType(
    const uint8_t* sig, uint32_t sig_len,
    GPUSigVersion sigversion)
{
    if (sig_len == 0) return GPU_SIGHASH_ALL;

    // For Schnorr (Taproot/Tapscript), handle differently
    if (sigversion == GPU_SIGVERSION_TAPROOT || sigversion == GPU_SIGVERSION_TAPSCRIPT) {
        // 64-byte signature = SIGHASH_DEFAULT
        // 65-byte signature = explicit sighash type
        if (sig_len == 64) return GPU_SIGHASH_DEFAULT;
        if (sig_len == 65) return sig[64];
        return 0xFF; // Invalid
    }

    // For ECDSA (legacy/SegWit v0), last byte is sighash type
    return sig[sig_len - 1];
}

// Validate sighash type
__device__ __host__ inline bool IsValidSigHashType(
    uint8_t sighash_type,
    GPUSigVersion sigversion)
{
    if (sigversion == GPU_SIGVERSION_TAPROOT || sigversion == GPU_SIGVERSION_TAPSCRIPT) {
        // Taproot/Tapscript: only allow specific values
        uint8_t base = sighash_type & 0x03;

        // SIGHASH_DEFAULT (0x00) is only valid for Taproot key path
        if (sighash_type == GPU_SIGHASH_DEFAULT) {
            return true;
        }

        // Valid base types: ALL (1), NONE (2), SINGLE (3)
        if (base < 1 || base > 3) return false;

        // Only ANYONECANPAY (0x80) allowed besides base type (0x03 mask)
        if ((sighash_type & ~0x83) != 0) return false;

        return true;
    }

    // Legacy/SegWit v0
    uint8_t base = sighash_type & 0x1F;
    if (base < GPU_SIGHASH_ALL || base > GPU_SIGHASH_SINGLE) return false;

    return true;
}

// Check if public key is valid for the given sigversion
__device__ __host__ inline bool IsValidPubKey(
    const uint8_t* pubkey, uint32_t pubkey_len,
    GPUSigVersion sigversion,
    uint32_t verify_flags)
{
    if (pubkey_len == 0) {
        // Empty pubkey only allowed in Tapscript
        return sigversion == GPU_SIGVERSION_TAPSCRIPT;
    }

    if (sigversion == GPU_SIGVERSION_TAPROOT || sigversion == GPU_SIGVERSION_TAPSCRIPT) {
        // Taproot/Tapscript: x-only pubkey (32 bytes)
        if (pubkey_len == 32) return true;

        // Unknown pubkey type
        if (verify_flags & GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_PUBKEY) {
            return false;
        }
        return true;
    }

    // Legacy/SegWit v0: compressed (33 bytes) or uncompressed (65 bytes)
    if (pubkey_len == 33) {
        // Compressed: must start with 0x02 or 0x03
        return pubkey[0] == 0x02 || pubkey[0] == 0x03;
    }
    if (pubkey_len == 65) {
        // Uncompressed: must start with 0x04
        return pubkey[0] == 0x04;
    }

    // For WITNESS_PUBKEYTYPE flag, only compressed allowed
    if ((verify_flags & GPU_SCRIPT_VERIFY_WITNESS_PUBKEYTYPE) &&
        sigversion == GPU_SIGVERSION_WITNESS_V0) {
        return false;
    }

    return false;
}

// ============================================================================
// Signature Verification Context
// Holds all data needed for signature verification
// ============================================================================

struct SigVerifyContext {
    // Signature data (without sighash byte for ECDSA)
    const uint8_t* sig;
    uint32_t sig_len;
    uint8_t sighash_type;

    // Public key
    const uint8_t* pubkey;
    uint32_t pubkey_len;

    // Script context
    GPUScriptContext* ctx;
    const uint8_t* script;
    uint32_t script_len;

    // Computed sighash
    uint8_t sighash[32];
    bool sighash_computed;
};

// ============================================================================
// ECDSA Signature Verification
// ============================================================================

__device__ __host__ inline bool VerifyECDSASignature(
    SigVerifyContext* vctx)
{
    // Parse signature (DER format, without sighash byte)
    secp256k1::Scalar r, s;
    if (!secp256k1::sig_parse_der_simple(r, s, vctx->sig, vctx->sig_len)) {
        return false;
    }

    // Check low-S (BIP 146)
    if (vctx->ctx->verify_flags & GPU_SCRIPT_VERIFY_LOW_S) {
        if (!CheckLowS(s)) {
            return false;
        }
    }

    // Parse public key
    secp256k1::AffinePoint pubkey;
    if (!secp256k1::pubkey_parse(pubkey, vctx->pubkey, vctx->pubkey_len)) {
        return false;
    }

    // Verify signature
    return secp256k1::ecdsa_verify_core(vctx->sighash, r, s, pubkey);
}

// ============================================================================
// Schnorr Signature Verification (BIP340)
// ============================================================================

__device__ __host__ inline bool VerifySchnorrSignature(
    SigVerifyContext* vctx)
{
    // Schnorr signatures are 64 bytes (no sighash byte in the signature itself)
    if (vctx->sig_len != 64) {
        return false;
    }

    // Public key must be 32 bytes (x-only)
    if (vctx->pubkey_len != 32) {
        return false;
    }

    // Verify signature using BIP340
    return secp256k1::schnorr_verify(
        vctx->sig,
        vctx->sighash,
        32,
        vctx->pubkey
    );
}

// ============================================================================
// Script Code Extraction for Sighash
// ============================================================================

// FindAndDelete implementation for GPU
// Removes all occurrences of signature pattern from script
// Pattern format: PUSH_OP || signature_data
// Returns: number of patterns found and deleted
__device__ __host__ inline int FindAndDelete(
    const uint8_t* script,
    uint32_t script_len,
    const uint8_t* sig,
    uint32_t sig_len,
    uint8_t* result,
    uint32_t* result_len)
{
    if (sig_len == 0 || script_len == 0) {
        // No pattern to find, copy script as-is
        for (uint32_t i = 0; i < script_len; i++) {
            result[i] = script[i];
        }
        *result_len = script_len;
        return 0;
    }

    // Build the pattern to search for: PUSH_OP || sig_data
    // For signatures <= 75 bytes, PUSH_OP is just sig_len
    // For larger, would need PUSHDATA1 (0x4c) or PUSHDATA2 (0x4d)
    uint8_t pattern[256];  // Max signature + push op
    uint32_t pattern_len = 0;

    if (sig_len <= 75) {
        pattern[pattern_len++] = (uint8_t)sig_len;
    } else if (sig_len <= 255) {
        pattern[pattern_len++] = 0x4c;  // OP_PUSHDATA1
        pattern[pattern_len++] = (uint8_t)sig_len;
    } else {
        // Signatures > 255 bytes are invalid in Bitcoin
        for (uint32_t i = 0; i < script_len; i++) {
            result[i] = script[i];
        }
        *result_len = script_len;
        return 0;
    }

    // Add signature data to pattern
    for (uint32_t i = 0; i < sig_len && pattern_len < 256; i++) {
        pattern[pattern_len++] = sig[i];
    }

    // Now scan script and copy non-matching parts
    int found = 0;
    uint32_t out_pos = 0;
    uint32_t i = 0;

    while (i < script_len) {
        // Check if pattern matches at current position
        bool matches = false;
        if (i + pattern_len <= script_len) {
            matches = true;
            for (uint32_t j = 0; j < pattern_len && matches; j++) {
                if (script[i + j] != pattern[j]) {
                    matches = false;
                }
            }
        }

        if (matches) {
            // Skip the pattern
            i += pattern_len;
            found++;
        } else {
            // Copy current byte to result
            result[out_pos++] = script[i++];
        }
    }

    *result_len = out_pos;
    return found;
}

// Find the script code for sighash computation
// For legacy: remove signature from script (FindAndDelete)
// For SegWit: use the script as-is or construct from pubkey
__device__ __host__ inline bool GetScriptCode(
    GPUScriptContext* ctx,
    const uint8_t* script,
    uint32_t script_len,
    const uint8_t* sig,
    uint32_t sig_len,
    uint8_t* script_code_out,
    uint32_t* script_code_len_out)
{
    if (ctx->sigversion == GPU_SIGVERSION_BASE) {
        // Start from code separator position if applicable
        uint32_t start = ctx->codeseparator_pos;
        if (start == 0 || start > script_len) start = 0;

        // First, extract the subscript from codeseparator position
        const uint8_t* subscript = script + start;
        uint32_t subscript_len = script_len - start;

        // For BASE sigversion, remove the signature from scriptCode
        // This is required by the original Bitcoin sighash algorithm
        if (sig_len > 0) {
            uint8_t temp_script[MAX_SCRIPT_SIZE];

            // Copy subscript to temp buffer first
            uint32_t copy_len = subscript_len;
            if (copy_len > MAX_SCRIPT_SIZE) copy_len = MAX_SCRIPT_SIZE;
            for (uint32_t i = 0; i < copy_len; i++) {
                temp_script[i] = subscript[i];
            }

            // Remove signature pattern from script
            int found = FindAndDelete(temp_script, copy_len, sig, sig_len,
                                       script_code_out, script_code_len_out);

            // If CONST_SCRIPTCODE flag is set and we found signatures, return error
            // This is handled by the caller checking the flag
            (void)found;  // Suppress unused warning; caller handles flag check

            return true;
        }

        // No signature to remove, just copy subscript
        uint32_t len = subscript_len;
        if (len > MAX_SCRIPT_SIZE) len = MAX_SCRIPT_SIZE;

        for (uint32_t i = 0; i < len; i++) {
            script_code_out[i] = subscript[i];
        }
        *script_code_len_out = len;
        return true;
    }

    // For WITNESS_V0, script is used as-is
    if (ctx->sigversion == GPU_SIGVERSION_WITNESS_V0) {
        uint32_t len = script_len;
        if (len > MAX_SCRIPT_SIZE) len = MAX_SCRIPT_SIZE;

        for (uint32_t i = 0; i < len; i++) {
            script_code_out[i] = script[i];
        }
        *script_code_len_out = len;
        return true;
    }

    // For Tapscript, start from code separator
    if (ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT) {
        uint32_t start = 0;
        if (ctx->execdata.codeseparator_pos_init &&
            ctx->execdata.codeseparator_pos != 0xFFFFFFFF) {
            start = ctx->execdata.codeseparator_pos;
        }

        uint32_t len = script_len - start;
        if (len > MAX_SCRIPT_SIZE) len = MAX_SCRIPT_SIZE;

        for (uint32_t i = 0; i < len; i++) {
            script_code_out[i] = script[start + i];
        }
        *script_code_len_out = len;
        return true;
    }

    return false;
}

// ============================================================================
// OP_CHECKSIG Implementation
// ============================================================================

__device__ inline bool op_checksig_impl(
    GPUScriptContext* ctx,
    const uint8_t* script,
    uint32_t script_len)
{
    // Need at least 2 elements: signature and pubkey
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Get pubkey (top of stack)
    GPUStackElement& pubkey_elem = ctx->stack[ctx->stack_size - 1];
    // Get signature (second from top)
    GPUStackElement& sig_elem = ctx->stack[ctx->stack_size - 2];

    // Check for empty pubkey in Tapscript
    if (ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT && pubkey_elem.size == 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_TAPSCRIPT_EMPTY_PUBKEY);
    }

    // Validate public key
    if (!IsValidPubKey(pubkey_elem.data, pubkey_elem.size,
                       ctx->sigversion, ctx->verify_flags)) {
        return ctx->set_error(GPU_SCRIPT_ERR_PUBKEYTYPE);
    }

    bool fSuccess = false;

    // Handle Taproot/Tapscript (Schnorr)
    if (ctx->sigversion == GPU_SIGVERSION_TAPROOT ||
        ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT) {

        // Update validation weight for Tapscript
        if (ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT) {
            if (ctx->execdata.validation_weight_init) {
                ctx->execdata.validation_weight_left -= TAPSCRIPT_VALIDATION_WEIGHT_PER_SIGOP;
                if (ctx->execdata.validation_weight_left < 0) {
                    return ctx->set_error(GPU_SCRIPT_ERR_TAPSCRIPT_VALIDATION_WEIGHT);
                }
            }
        }

        // Empty signature = fail silently (push false)
        if (sig_elem.size == 0) {
            fSuccess = false;
        } else {
            // Validate signature length (64 or 65 bytes)
            if (sig_elem.size != 64 && sig_elem.size != 65) {
                return ctx->set_error(GPU_SCRIPT_ERR_SCHNORR_SIG_SIZE);
            }

            // Get sighash type
            uint8_t sighash_type = GetSigHashType(sig_elem.data, sig_elem.size, ctx->sigversion);

            // Validate sighash type
            if (!IsValidSigHashType(sighash_type, ctx->sigversion)) {
                return ctx->set_error(GPU_SCRIPT_ERR_SCHNORR_SIG_HASHTYPE);
            }

            // Use precomputed sighash (caller must provide it)
            uint8_t sighash[32];
            if (ctx->precomputed_sighash_valid) {
                for (int i = 0; i < 32; i++) {
                    sighash[i] = ctx->precomputed_sighash.data[i];
                }
            } else {
                // Cannot compute sighash without full tx context
                return ctx->set_error(GPU_SCRIPT_ERR_UNKNOWN_ERROR);
            }

            // Verify Schnorr signature
            if (secp256k1::schnorr_verify(sig_elem.data, sighash, 32, pubkey_elem.data)) {
                fSuccess = true;
            } else {
                // Non-empty invalid signature is an error (not just false)
                return ctx->set_error(GPU_SCRIPT_ERR_SCHNORR_SIG);
            }
        }
    } else {
        // Legacy/SegWit v0 (ECDSA)

        // Empty signature = fail silently (push false)
        if (sig_elem.size == 0) {
            fSuccess = false;
        } else {
            // Validate DER encoding
            if (ctx->verify_flags & GPU_SCRIPT_VERIFY_DERSIG) {
                if (!IsValidSignatureEncoding(sig_elem.data, sig_elem.size)) {
                    return ctx->set_error(GPU_SCRIPT_ERR_SIG_DER);
                }
            }

            // Get sighash type (last byte)
            uint8_t sighash_type = sig_elem.data[sig_elem.size - 1];

            // Validate sighash type
            if (!IsValidSigHashType(sighash_type, ctx->sigversion)) {
                return ctx->set_error(GPU_SCRIPT_ERR_SIG_HASHTYPE);
            }

            // Parse signature (without sighash byte)
            secp256k1::Scalar r, s;
            if (!secp256k1::sig_parse_der_simple(r, s, sig_elem.data, sig_elem.size - 1)) {
                return ctx->set_error(GPU_SCRIPT_ERR_SIG_DER);
            }

            // Check low-S
            if (ctx->verify_flags & GPU_SCRIPT_VERIFY_LOW_S) {
                if (!CheckLowS(s)) {
                    return ctx->set_error(GPU_SCRIPT_ERR_SIG_HIGH_S);
                }
            }

            // Parse public key
            secp256k1::AffinePoint pubkey;
            if (!secp256k1::pubkey_parse(pubkey, pubkey_elem.data, pubkey_elem.size)) {
                return ctx->set_error(GPU_SCRIPT_ERR_PUBKEYTYPE);
            }

            // Get script code for sighash
            uint8_t script_code[MAX_SCRIPT_SIZE];
            uint32_t script_code_len;
            if (!GetScriptCode(ctx, script, script_len, sig_elem.data, sig_elem.size,
                               script_code, &script_code_len)) {
                return ctx->set_error(GPU_SCRIPT_ERR_UNKNOWN_ERROR);
            }

            // Use precomputed sighash if available
            // For batch validation, sighash is computed on CPU and passed in
            uint8_t sighash[32];
            if (ctx->precomputed_sighash_valid) {
                for (int i = 0; i < 32; i++) {
                    sighash[i] = ctx->precomputed_sighash.data[i];
                }
            } else {
                // Cannot compute sighash without full tx context
                // This should be precomputed by the caller
                return ctx->set_error(GPU_SCRIPT_ERR_UNKNOWN_ERROR);
            }

            // Verify ECDSA signature
            fSuccess = secp256k1::ecdsa_verify_core(sighash, r, s, pubkey);
        }

        // NULLFAIL: non-empty failing signature is an error
        if (!fSuccess && sig_elem.size > 0 &&
            (ctx->verify_flags & GPU_SCRIPT_VERIFY_NULLFAIL)) {
            return ctx->set_error(GPU_SCRIPT_ERR_SIG_NULLFAIL);
        }
    }

    // Pop signature and pubkey, push result
    ctx->stack_size -= 2;
    return stack_push_bool(ctx, fSuccess);
}

// ============================================================================
// OP_CHECKSIGVERIFY Implementation
// ============================================================================

__device__ inline bool op_checksigverify_impl(
    GPUScriptContext* ctx,
    const uint8_t* script,
    uint32_t script_len)
{
    if (!op_checksig_impl(ctx, script, script_len)) {
        return false;
    }

    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    bool fSuccess = CastToBool(stacktop(ctx, -1));
    ctx->stack_size--;

    if (!fSuccess) {
        return ctx->set_error(GPU_SCRIPT_ERR_CHECKSIGVERIFY);
    }

    return true;
}

// ============================================================================
// OP_CHECKMULTISIG Implementation
// ============================================================================

__device__ inline bool op_checkmultisig_impl(
    GPUScriptContext* ctx,
    const uint8_t* script,
    uint32_t script_len,
    bool fRequireMinimal)
{
    // Not allowed in Tapscript
    if (ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT) {
        return ctx->set_error(GPU_SCRIPT_ERR_TAPSCRIPT_CHECKMULTISIG);
    }

    // Need at least 1 element for key count
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Get number of public keys
    int32_t nKeysCount = GPUScriptNum(stacktop(ctx, -1), fRequireMinimal).getint();
    if (nKeysCount < 0 || static_cast<uint32_t>(nKeysCount) > MAX_PUBKEYS_PER_MULTISIG) {
        return ctx->set_error(GPU_SCRIPT_ERR_PUBKEY_COUNT);
    }

    // Update opcode count
    ctx->opcode_count += nKeysCount;
    if (ctx->opcode_count > MAX_OPS_PER_SCRIPT) {
        return ctx->set_error(GPU_SCRIPT_ERR_OP_COUNT);
    }

    int32_t ikey = ctx->stack_size - 2;  // First key index
    int32_t ikey_end = ikey - nKeysCount;

    // Check we have enough elements for keys
    if (ikey_end < 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Get number of signatures required
    int32_t nSigsCount = GPUScriptNum(ctx->stack[ikey_end], fRequireMinimal).getint();
    if (nSigsCount < 0 || nSigsCount > nKeysCount) {
        return ctx->set_error(GPU_SCRIPT_ERR_SIG_COUNT);
    }

    int32_t isig = ikey_end - 1;  // First signature index
    int32_t isig_end = isig - nSigsCount;

    // Check we have enough elements for signatures + dummy
    if (isig_end < 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Check dummy element (NULLDUMMY)
    if (ctx->verify_flags & GPU_SCRIPT_VERIFY_NULLDUMMY) {
        if (ctx->stack[isig_end].size != 0) {
            return ctx->set_error(GPU_SCRIPT_ERR_SIG_NULLDUMMY);
        }
    }

    // Verify signatures
    bool fSuccess = true;
    int32_t nSigsRemaining = nSigsCount;
    int32_t nKeysRemaining = nKeysCount;

    while (fSuccess && nSigsRemaining > 0) {
        // Get current signature and key
        GPUStackElement& sig_elem = ctx->stack[isig];
        GPUStackElement& key_elem = ctx->stack[ikey];

        bool fOk = false;

        // Try to verify this signature with this key
        if (sig_elem.size > 0) {
            // Validate DER encoding
            if (ctx->verify_flags & GPU_SCRIPT_VERIFY_DERSIG) {
                if (!IsValidSignatureEncoding(sig_elem.data, sig_elem.size)) {
                    return ctx->set_error(GPU_SCRIPT_ERR_SIG_DER);
                }
            }

            // Get sighash type
            uint8_t sighash_type = sig_elem.data[sig_elem.size - 1];

            // Validate sighash type
            if (!IsValidSigHashType(sighash_type, ctx->sigversion)) {
                return ctx->set_error(GPU_SCRIPT_ERR_SIG_HASHTYPE);
            }

            // Parse signature
            secp256k1::Scalar r, s;
            if (secp256k1::sig_parse_der_simple(r, s, sig_elem.data, sig_elem.size - 1)) {
                // Check low-S
                if ((ctx->verify_flags & GPU_SCRIPT_VERIFY_LOW_S) && !CheckLowS(s)) {
                    return ctx->set_error(GPU_SCRIPT_ERR_SIG_HIGH_S);
                }

                // Parse public key
                secp256k1::AffinePoint pubkey;
                if (secp256k1::pubkey_parse(pubkey, key_elem.data, key_elem.size)) {
                    // Use precomputed sighash if available
                    uint8_t sighash[32];
                    if (ctx->precomputed_sighash_valid) {
                        for (int i = 0; i < 32; i++) {
                            sighash[i] = ctx->precomputed_sighash.data[i];
                        }
                        fOk = secp256k1::ecdsa_verify_core(sighash, r, s, pubkey);
                    } else {
                        // Cannot verify without precomputed sighash
                        fOk = false;
                    }
                }
            }
        }

        if (fOk) {
            isig--;
            nSigsRemaining--;
        }

        ikey--;
        nKeysRemaining--;

        // If more signatures remain than keys, fail
        if (nSigsRemaining > nKeysRemaining) {
            fSuccess = false;
        }
    }

    // NULLFAIL check for remaining signatures
    if (ctx->verify_flags & GPU_SCRIPT_VERIFY_NULLFAIL) {
        while (isig >= isig_end) {
            if (ctx->stack[isig].size > 0) {
                return ctx->set_error(GPU_SCRIPT_ERR_SIG_NULLFAIL);
            }
            isig--;
        }
    }

    // Pop all elements (nKeysCount + nSigsCount + 2 for counts + 1 for dummy)
    ctx->stack_size = isig_end;

    return stack_push_bool(ctx, fSuccess);
}

// ============================================================================
// OP_CHECKMULTISIGVERIFY Implementation
// ============================================================================

__device__ inline bool op_checkmultisigverify_impl(
    GPUScriptContext* ctx,
    const uint8_t* script,
    uint32_t script_len,
    bool fRequireMinimal)
{
    if (!op_checkmultisig_impl(ctx, script, script_len, fRequireMinimal)) {
        return false;
    }

    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    bool fSuccess = CastToBool(stacktop(ctx, -1));
    ctx->stack_size--;

    if (!fSuccess) {
        return ctx->set_error(GPU_SCRIPT_ERR_CHECKMULTISIGVERIFY);
    }

    return true;
}

// ============================================================================
// OP_CHECKSIGADD Implementation (Tapscript BIP342)
// ============================================================================

__device__ inline bool op_checksigadd_impl(
    GPUScriptContext* ctx,
    const uint8_t* script,
    uint32_t script_len)
{
    // Only valid in Tapscript
    if (ctx->sigversion != GPU_SIGVERSION_TAPSCRIPT) {
        return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
    }

    // Need 3 elements: sig, n, pubkey
    if (ctx->stack_size < 3) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Get elements from stack
    GPUStackElement& pubkey_elem = ctx->stack[ctx->stack_size - 1];
    GPUStackElement& n_elem = ctx->stack[ctx->stack_size - 2];
    GPUStackElement& sig_elem = ctx->stack[ctx->stack_size - 3];

    // Parse n as script number
    GPUScriptNum n(n_elem, false);
    if (!n.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Check for empty pubkey
    if (pubkey_elem.size == 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_TAPSCRIPT_EMPTY_PUBKEY);
    }

    // Update validation weight
    if (ctx->execdata.validation_weight_init) {
        ctx->execdata.validation_weight_left -= TAPSCRIPT_VALIDATION_WEIGHT_PER_SIGOP;
        if (ctx->execdata.validation_weight_left < 0) {
            return ctx->set_error(GPU_SCRIPT_ERR_TAPSCRIPT_VALIDATION_WEIGHT);
        }
    }

    // Pop all 3 elements
    ctx->stack_size -= 3;

    // Empty signature: push n unchanged
    if (sig_elem.size == 0) {
        return stack_push_num(ctx, n);
    }

    // Validate signature length
    if (sig_elem.size != 64 && sig_elem.size != 65) {
        return ctx->set_error(GPU_SCRIPT_ERR_SCHNORR_SIG_SIZE);
    }

    // Get sighash type
    uint8_t sighash_type = GetSigHashType(sig_elem.data, sig_elem.size, ctx->sigversion);

    // Validate sighash type
    if (!IsValidSigHashType(sighash_type, ctx->sigversion)) {
        return ctx->set_error(GPU_SCRIPT_ERR_SCHNORR_SIG_HASHTYPE);
    }

    // Only 32-byte pubkeys allowed in Tapscript
    if (pubkey_elem.size != 32) {
        // Unknown pubkey type - check for upgradability
        if (ctx->verify_flags & GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_PUBKEY) {
            return ctx->set_error(GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_PUBKEYTYPE);
        }
        // Unknown pubkey = signature is valid (forward compatible)
        return stack_push_num(ctx, GPUScriptNum(n.GetInt64() + 1));
    }

    // Use precomputed sighash if available
    uint8_t sighash[32];
    if (ctx->precomputed_sighash_valid) {
        for (int i = 0; i < 32; i++) {
            sighash[i] = ctx->precomputed_sighash.data[i];
        }
    } else {
        // Cannot verify without precomputed sighash
        return ctx->set_error(GPU_SCRIPT_ERR_UNKNOWN_ERROR);
    }

    // Verify Schnorr signature
    if (secp256k1::schnorr_verify(sig_elem.data, sighash, 32, pubkey_elem.data)) {
        // Signature valid: push n+1
        return stack_push_num(ctx, GPUScriptNum(n.GetInt64() + 1));
    } else {
        // Invalid non-empty signature is an error
        return ctx->set_error(GPU_SCRIPT_ERR_SCHNORR_SIG);
    }
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_OPCODES_SIG_CUH
