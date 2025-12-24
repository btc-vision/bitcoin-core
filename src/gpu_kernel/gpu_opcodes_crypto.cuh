// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_OPCODES_CRYPTO_CUH
#define BITCOIN_GPU_KERNEL_GPU_OPCODES_CRYPTO_CUH

#include "gpu_script_types.cuh"
#include "gpu_script_stack.cuh"
#include "gpu_hash.cuh"

namespace gpu {

// ============================================================================
// Crypto Opcode Implementations
// These use the GPU hash functions from gpu_hash.cuh
// ============================================================================

// OP_RIPEMD160: Hash top element with RIPEMD-160
// (in -- hash)
__device__ inline bool op_ripemd160(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& elem = stacktop(ctx, -1);

    // Compute RIPEMD-160 hash
    uint8_t hash[20];
    ripemd160(elem.data, elem.size, hash);

    // Replace top element with hash
    memcpy(elem.data, hash, 20);
    elem.size = 20;

    return true;
}

// OP_SHA1: Hash top element with SHA-1
// (in -- hash)
__device__ inline bool op_sha1(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& elem = stacktop(ctx, -1);

    // Compute SHA-1 hash
    uint8_t hash[20];
    sha1(elem.data, elem.size, hash);

    // Replace top element with hash
    memcpy(elem.data, hash, 20);
    elem.size = 20;

    return true;
}

// OP_SHA256: Hash top element with SHA-256
// (in -- hash)
__device__ inline bool op_sha256(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& elem = stacktop(ctx, -1);

    // Compute SHA-256 hash
    uint8_t hash[32];
    sha256(elem.data, elem.size, hash);

    // Replace top element with hash
    memcpy(elem.data, hash, 32);
    elem.size = 32;

    return true;
}

// OP_HASH160: Hash top element with SHA-256 then RIPEMD-160
// (in -- hash)
__device__ inline bool op_hash160(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& elem = stacktop(ctx, -1);

    // Compute HASH160 (SHA256 + RIPEMD160)
    uint8_t hash[20];
    hash160(elem.data, elem.size, hash);

    // Replace top element with hash
    memcpy(elem.data, hash, 20);
    elem.size = 20;

    return true;
}

// OP_HASH256: Hash top element with double SHA-256
// (in -- hash)
__device__ inline bool op_hash256(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& elem = stacktop(ctx, -1);

    // Compute double SHA-256
    uint8_t hash[32];
    sha256d(elem.data, elem.size, hash);

    // Replace top element with hash
    memcpy(elem.data, hash, 32);
    elem.size = 32;

    return true;
}

// OP_CODESEPARATOR: Mark the current position in the script
// This is used for signature operations to determine what part of the script to sign
__device__ inline bool op_codeseparator(GPUScriptContext* ctx, uint32_t current_pos)
{
    // For BASE sigversion with CONST_SCRIPTCODE flag, this is an error
    if (ctx->sigversion == GPU_SIGVERSION_BASE &&
        (ctx->verify_flags & GPU_SCRIPT_VERIFY_CONST_SCRIPTCODE)) {
        return ctx->set_error(GPU_SCRIPT_ERR_OP_CODESEPARATOR);
    }

    // Update the codeseparator position
    ctx->codeseparator_pos = current_pos;

    // For Tapscript, update execdata
    if (ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT) {
        ctx->execdata.codeseparator_pos = current_pos;
        ctx->execdata.codeseparator_pos_init = true;
    }

    return true;
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_OPCODES_CRYPTO_CUH
