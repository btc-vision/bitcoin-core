// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "gpu_batch_validator.h"
#include "gpu_eval_script.cuh"
#include "gpu_hash.cuh"
#include "gpu_ecdsa_verify.cuh"
#include "gpu_schnorr_verify.cuh"
#include "gpu_sighash.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <cstdio>

// Simple logging macros for CUDA code
#define GPU_LOG_ERROR(fmt, ...) fprintf(stderr, "[GPU ERROR] " fmt "\n", ##__VA_ARGS__)
#define GPU_LOG_WARNING(fmt, ...) fprintf(stderr, "[GPU WARN] " fmt "\n", ##__VA_ARGS__)
#define GPU_LOG_INFO(fmt, ...) fprintf(stdout, "[GPU INFO] " fmt "\n", ##__VA_ARGS__)

namespace gpu {

// Device-only version of IdentifyScriptType for kernel use
// Matches Bitcoin Core's TxoutType detection in solver.cpp
__device__ inline ScriptType IdentifyScriptTypeDevice(const uint8_t* script, uint32_t size) {
    if (!script || size == 0) return SCRIPT_TYPE_UNKNOWN;

    // P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
    if (size == 25 && script[0] == 0x76 && script[1] == 0xa9 &&
        script[2] == 0x14 && script[23] == 0x88 && script[24] == 0xac) {
        return SCRIPT_TYPE_P2PKH;
    }

    // P2SH: OP_HASH160 <20 bytes> OP_EQUAL
    if (size == 23 && script[0] == 0xa9 && script[1] == 0x14 && script[22] == 0x87) {
        return SCRIPT_TYPE_P2SH;
    }

    // P2WPKH: OP_0 <20 bytes>
    if (size == 22 && script[0] == 0x00 && script[1] == 0x14) {
        return SCRIPT_TYPE_P2WPKH;
    }

    // P2WSH: OP_0 <32 bytes>
    if (size == 34 && script[0] == 0x00 && script[1] == 0x20) {
        return SCRIPT_TYPE_P2WSH;
    }

    // P2TR: OP_1 <32 bytes>
    if (size == 34 && script[0] == 0x51 && script[1] == 0x20) {
        return SCRIPT_TYPE_P2TR;
    }

    // P2PK: <33 or 65 byte compressed/uncompressed pubkey> OP_CHECKSIG
    // Compressed: 0x21 <33 bytes> 0xac
    // Uncompressed: 0x41 <65 bytes> 0xac
    if ((size == 35 && script[0] == 0x21 && script[34] == 0xac &&
         (script[1] == 0x02 || script[1] == 0x03)) ||  // Compressed pubkey
        (size == 67 && script[0] == 0x41 && script[66] == 0xac &&
         script[1] == 0x04)) {  // Uncompressed pubkey
        return SCRIPT_TYPE_P2PK;
    }

    // NULL_DATA (OP_RETURN): OP_RETURN <optional data up to 80 bytes>
    if (size >= 1 && script[0] == 0x6a) {  // OP_RETURN
        return SCRIPT_TYPE_NULL_DATA;
    }

    // WITNESS_UNKNOWN: OP_N <2-40 bytes> where N is 2-16 (witness version 2+)
    // OP_2 = 0x52, OP_16 = 0x60
    if (size >= 4 && size <= 42 && script[0] >= 0x52 && script[0] <= 0x60) {
        uint8_t push_size = script[1];
        // Check it's a valid witness program (2-40 bytes, direct push)
        if (push_size >= 2 && push_size <= 40 && size == (uint32_t)(2 + push_size)) {
            return SCRIPT_TYPE_WITNESS_UNKNOWN;
        }
    }

    // MULTISIG: OP_M <pubkey1> ... <pubkeyN> OP_N OP_CHECKMULTISIG
    // Check for OP_CHECKMULTISIG at end (0xae) or OP_CHECKMULTISIGVERIFY (0xaf)
    if (size >= 37 && (script[size-1] == 0xae || script[size-1] == 0xaf)) {
        // First byte should be OP_1 through OP_16 (required sigs)
        if (script[0] >= 0x51 && script[0] <= 0x60) {
            // Second to last byte should be OP_1 through OP_16 (total keys)
            if (script[size-2] >= 0x51 && script[size-2] <= 0x60) {
                return SCRIPT_TYPE_MULTISIG;
            }
        }
    }

    return SCRIPT_TYPE_NONSTANDARD;
}

// ============================================================================
// GPUBatchValidator Implementation
// ============================================================================

GPUBatchValidator::GPUBatchValidator()
    : m_initialized(false)
    , m_batch_active(false)
    , m_job_count(0)
    , m_max_jobs(0)
    , m_scriptpubkey_used(0)
    , m_scriptsig_used(0)
    , m_witness_used(0)
    , m_scriptpubkey_max(0)
    , m_scriptsig_max(0)
    , m_witness_max(0)
    , d_jobs(nullptr)
    , d_scriptpubkey_blob(nullptr)
    , d_scriptsig_blob(nullptr)
    , d_witness_blob(nullptr)
    , d_contexts(nullptr)
    , d_tx_contexts(nullptr)
    , m_utxo_set(nullptr)
{
}

GPUBatchValidator::~GPUBatchValidator()
{
    Shutdown();
}

bool GPUBatchValidator::Initialize(size_t max_jobs, size_t script_blob_size, size_t witness_blob_size)
{
    if (m_initialized) {
        return true;
    }

    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GPU_LOG_ERROR("No CUDA devices available for batch validation");
        return false;
    }

    // Query available GPU memory to dynamically size max_jobs
    size_t free_mem = 0, total_mem = 0;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        GPU_LOG_WARNING("Could not query GPU memory, using defaults");
        free_mem = 1024ULL * 1024 * 1024;  // Assume 1GB if query fails
    }

    // Reserve 100MB for other allocations and overhead
    const size_t OVERHEAD_RESERVE = 100ULL * 1024 * 1024;
    size_t available = (free_mem > OVERHEAD_RESERVE) ? (free_mem - OVERHEAD_RESERVE) : 0;

    // Calculate how many contexts can fit
    // Each GPUScriptContext is ~1.05MB (CONTEXT_SIZE_BYTES)
    // Plus we need space for: jobs array, script blobs, witness blob, tx contexts
    size_t per_job_overhead = sizeof(ScriptValidationJob) + sizeof(TxContext) +
                              256;  // Average script data per job
    size_t fixed_overhead = script_blob_size + witness_blob_size;

    size_t memory_per_job = CONTEXT_SIZE_BYTES + per_job_overhead;
    size_t max_possible_jobs = 0;
    if (available > fixed_overhead) {
        max_possible_jobs = (available - fixed_overhead) / memory_per_job;
    }

    // Apply limits
    if (max_possible_jobs < MIN_MAX_JOBS) {
        GPU_LOG_ERROR("Insufficient GPU memory for batch validation: %zu MB free, need at least %zu MB",
                      free_mem / (1024 * 1024),
                      (MIN_MAX_JOBS * memory_per_job + fixed_overhead) / (1024 * 1024));
        return false;
    }

    // Cap at requested max_jobs or available memory, whichever is smaller
    m_max_jobs = (max_possible_jobs < max_jobs) ? max_possible_jobs : max_jobs;

    GPU_LOG_INFO("GPU memory: %zu MB free, allocating batch validator for %zu jobs (%.1f MB for contexts)",
                 free_mem / (1024 * 1024), m_max_jobs,
                 (m_max_jobs * CONTEXT_SIZE_BYTES) / (1024.0 * 1024.0));

    m_scriptpubkey_max = script_blob_size / 2;
    m_scriptsig_max = script_blob_size / 2;
    m_witness_max = witness_blob_size;

    // Allocate host-side staging buffers
    try {
        m_jobs.resize(max_jobs);
        m_scriptpubkey_blob.resize(m_scriptpubkey_max);
        m_scriptsig_blob.resize(m_scriptsig_max);
        m_witness_blob.resize(m_witness_max);
        m_tx_contexts.resize(max_jobs);
    } catch (const std::bad_alloc&) {
        GPU_LOG_ERROR("Failed to allocate host memory for batch validator");
        return false;
    }

    // Allocate device memory
    if (!AllocateDeviceMemory()) {
        GPU_LOG_ERROR("Failed to allocate device memory for batch validator");
        return false;
    }

    m_initialized = true;
    GPU_LOG_INFO("GPU Batch Validator initialized: max_jobs=%zu, script_blob=%zuMB, witness_blob=%zuMB",
                 max_jobs, script_blob_size / (1024 * 1024), witness_blob_size / (1024 * 1024));

    return true;
}

void GPUBatchValidator::Shutdown()
{
    if (!m_initialized) return;

    FreeDeviceMemory();

    m_jobs.clear();
    m_scriptpubkey_blob.clear();
    m_scriptsig_blob.clear();
    m_witness_blob.clear();
    m_tx_contexts.clear();

    m_initialized = false;
    m_batch_active = false;
    m_job_count = 0;
}

bool GPUBatchValidator::AllocateDeviceMemory()
{
    cudaError_t err;

    // Allocate job array
    err = cudaMalloc(&d_jobs, m_max_jobs * sizeof(ScriptValidationJob));
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMalloc failed for jobs: %s", cudaGetErrorString(err));
        return false;
    }

    // Allocate script blobs
    err = cudaMalloc(&d_scriptpubkey_blob, m_scriptpubkey_max);
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMalloc failed for scriptpubkey blob: %s", cudaGetErrorString(err));
        FreeDeviceMemory();
        return false;
    }

    err = cudaMalloc(&d_scriptsig_blob, m_scriptsig_max);
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMalloc failed for scriptsig blob: %s", cudaGetErrorString(err));
        FreeDeviceMemory();
        return false;
    }

    err = cudaMalloc(&d_witness_blob, m_witness_max);
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMalloc failed for witness blob: %s", cudaGetErrorString(err));
        FreeDeviceMemory();
        return false;
    }

    // Allocate transaction contexts
    err = cudaMalloc(&d_tx_contexts, m_max_jobs * sizeof(TxContext));
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMalloc failed for tx contexts: %s", cudaGetErrorString(err));
        FreeDeviceMemory();
        return false;
    }

    // Allocate script execution contexts (spilled to global memory for large stack support)
    err = cudaMalloc(&d_contexts, m_max_jobs * sizeof(GPUScriptContext));
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMalloc failed for script contexts: %s", cudaGetErrorString(err));
        FreeDeviceMemory();
        return false;
    }

    return true;
}

void GPUBatchValidator::FreeDeviceMemory()
{
    if (d_jobs) { cudaFree(d_jobs); d_jobs = nullptr; }
    if (d_scriptpubkey_blob) { cudaFree(d_scriptpubkey_blob); d_scriptpubkey_blob = nullptr; }
    if (d_scriptsig_blob) { cudaFree(d_scriptsig_blob); d_scriptsig_blob = nullptr; }
    if (d_witness_blob) { cudaFree(d_witness_blob); d_witness_blob = nullptr; }
    if (d_contexts) { cudaFree(d_contexts); d_contexts = nullptr; }
    if (d_tx_contexts) { cudaFree(d_tx_contexts); d_tx_contexts = nullptr; }
}

void GPUBatchValidator::BeginBatch()
{
    if (!m_initialized) return;

    m_batch_active = true;
    m_job_count = 0;
    m_scriptpubkey_used = 0;
    m_scriptsig_used = 0;
    m_witness_used = 0;
}

void GPUBatchValidator::EndBatch()
{
    m_batch_active = false;
}

int GPUBatchValidator::QueueJob(
    uint32_t tx_index,
    uint32_t input_index,
    const uint8_t* scriptpubkey, uint32_t scriptpubkey_len,
    const uint8_t* scriptsig, uint32_t scriptsig_len,
    const uint8_t* witness, uint32_t witness_len, uint32_t witness_count,
    int64_t amount,
    uint32_t sequence,
    uint32_t verify_flags,
    GPUSigVersion sigversion,
    const uint8_t* sighash)
{
    if (!m_initialized || !m_batch_active) {
        return -1;
    }

    if (m_job_count >= m_max_jobs) {
        GPU_LOG_WARNING("Batch validator job queue full");
        return -1;
    }

    // Check space in script blobs
    if (m_scriptpubkey_used + scriptpubkey_len > m_scriptpubkey_max) {
        GPU_LOG_WARNING("ScriptPubKey blob full");
        return -1;
    }
    if (m_scriptsig_used + scriptsig_len > m_scriptsig_max) {
        GPU_LOG_WARNING("ScriptSig blob full");
        return -1;
    }
    if (m_witness_used + witness_len > m_witness_max) {
        GPU_LOG_WARNING("Witness blob full");
        return -1;
    }

    // Create job
    ScriptValidationJob& job = m_jobs[m_job_count];
    job.tx_index = tx_index;
    job.input_index = input_index;

    // Copy scriptPubKey
    job.scriptpubkey_offset = m_scriptpubkey_used;
    job.scriptpubkey_size = scriptpubkey_len;
    if (scriptpubkey && scriptpubkey_len > 0) {
        memcpy(m_scriptpubkey_blob.data() + m_scriptpubkey_used, scriptpubkey, scriptpubkey_len);
        m_scriptpubkey_used += scriptpubkey_len;
    }

    // Copy scriptSig
    job.scriptsig_offset = m_scriptsig_used;
    job.scriptsig_size = scriptsig_len;
    if (scriptsig && scriptsig_len > 0) {
        memcpy(m_scriptsig_blob.data() + m_scriptsig_used, scriptsig, scriptsig_len);
        m_scriptsig_used += scriptsig_len;
    }

    // Copy witness data
    job.witness_offset = m_witness_used;
    job.witness_count = witness_count;
    job.witness_total_size = witness_len;
    if (witness && witness_len > 0) {
        memcpy(m_witness_blob.data() + m_witness_used, witness, witness_len);
        m_witness_used += witness_len;
    }

    // Set transaction context
    job.amount = amount;
    job.sequence = sequence;
    job.verify_flags = verify_flags;
    job.sigversion = sigversion;

    // Set precomputed sighash if provided
    if (sighash) {
        job.sighash_valid = true;
        memcpy(job.sighash.data, sighash, 32);
    } else {
        job.sighash_valid = false;
    }

    // Initialize result fields
    job.error = GPU_SCRIPT_ERR_OK;
    job.validated = false;
    job.valid = false;

    return static_cast<int>(m_job_count++);
}

GPUSigVersion GPUBatchValidator::DetermineSigVersion(
    const uint8_t* scriptpubkey, uint32_t len,
    const uint8_t* witness, uint32_t witness_len) const
{
    if (!scriptpubkey || len == 0) {
        return GPU_SIGVERSION_BASE;
    }

    // Check for witness programs
    if (len >= 2) {
        uint8_t version = scriptpubkey[0];
        uint8_t push_len = scriptpubkey[1];

        // SegWit version 0
        if (version == 0x00) {
            if (push_len == 20 && len == 22) {
                return GPU_SIGVERSION_WITNESS_V0;  // P2WPKH
            }
            if (push_len == 32 && len == 34) {
                return GPU_SIGVERSION_WITNESS_V0;  // P2WSH
            }
        }

        // SegWit version 1 (Taproot)
        if (version == 0x51 && push_len == 32 && len == 34) {
            // Check if key-path or script-path based on witness
            if (witness && witness_len > 0) {
                // If witness has more than one element and last is control block, it's script path
                // For simplicity, assume key-path unless we detect script path
                return GPU_SIGVERSION_TAPROOT;
            }
            return GPU_SIGVERSION_TAPROOT;
        }
    }

    // Default to legacy
    return GPU_SIGVERSION_BASE;
}

bool GPUBatchValidator::CopyToDevice()
{
    cudaError_t err;

    // Copy jobs
    err = cudaMemcpy(d_jobs, m_jobs.data(), m_job_count * sizeof(ScriptValidationJob), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMemcpy failed for jobs: %s", cudaGetErrorString(err));
        return false;
    }

    // Copy script blobs
    if (m_scriptpubkey_used > 0) {
        err = cudaMemcpy(d_scriptpubkey_blob, m_scriptpubkey_blob.data(), m_scriptpubkey_used, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            GPU_LOG_ERROR("cudaMemcpy failed for scriptpubkey blob: %s", cudaGetErrorString(err));
            return false;
        }
    }

    if (m_scriptsig_used > 0) {
        err = cudaMemcpy(d_scriptsig_blob, m_scriptsig_blob.data(), m_scriptsig_used, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            GPU_LOG_ERROR("cudaMemcpy failed for scriptsig blob: %s", cudaGetErrorString(err));
            return false;
        }
    }

    if (m_witness_used > 0) {
        err = cudaMemcpy(d_witness_blob, m_witness_blob.data(), m_witness_used, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            GPU_LOG_ERROR("cudaMemcpy failed for witness blob: %s", cudaGetErrorString(err));
            return false;
        }
    }

    return true;
}

bool GPUBatchValidator::CopyFromDevice()
{
    cudaError_t err = cudaMemcpy(m_jobs.data(), d_jobs, m_job_count * sizeof(ScriptValidationJob), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("cudaMemcpy failed for results: %s", cudaGetErrorString(err));
        return false;
    }
    return true;
}

BatchValidationResult GPUBatchValidator::ValidateBatch()
{
    BatchValidationResult result = {};
    result.total_jobs = m_job_count;

    if (!m_initialized || m_job_count == 0) {
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Copy data to device
    if (!CopyToDevice()) {
        GPU_LOG_ERROR("Failed to copy batch data to device");
        result.skipped_count = m_job_count;
        return result;
    }

    auto copy_done = std::chrono::high_resolution_clock::now();
    result.setup_time_ms = std::chrono::duration<double, std::milli>(copy_done - start_time).count();

    // Calculate grid dimensions
    int threads_per_block = 256;
    int num_blocks = (m_job_count + threads_per_block - 1) / threads_per_block;

    GPU_LOG_INFO("Kernel launch: blocks=%d, threads=%d, jobs=%zu, d_jobs=%p",
                 num_blocks, threads_per_block, m_job_count, (void*)d_jobs);

    // Launch validation kernel
    BatchValidateScriptsKernel<<<num_blocks, threads_per_block>>>(
        d_jobs,
        d_scriptpubkey_blob,
        d_scriptsig_blob,
        d_witness_blob,
        d_contexts,
        m_job_count
    );

    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        GPU_LOG_ERROR("Kernel launch failed: %s", cudaGetErrorString(launch_err));
        result.skipped_count = m_job_count;
        return result;
    }

    // Wait for completion
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        GPU_LOG_ERROR("Kernel execution failed: %s", cudaGetErrorString(err));
        result.skipped_count = m_job_count;
        return result;
    }

    auto kernel_done = std::chrono::high_resolution_clock::now();
    result.gpu_time_ms = std::chrono::duration<double, std::milli>(kernel_done - copy_done).count();

    // Copy results back
    if (!CopyFromDevice()) {
        GPU_LOG_ERROR("Failed to copy results from device");
        result.skipped_count = m_job_count;
        return result;
    }

    // Debug: Print first job's result details
    if (m_job_count > 0) {
        const ScriptValidationJob& first_job = m_jobs[0];
        GPU_LOG_INFO("Job 0 debug: validated=%d, valid=%d, error=%d, sigversion=%d",
                     (int)first_job.validated, (int)first_job.valid,
                     (int)first_job.error, (int)first_job.sigversion);
    }

    // Analyze results
    for (size_t i = 0; i < m_job_count; i++) {
        const ScriptValidationJob& job = m_jobs[i];

        if (job.validated) {
            result.validated_count++;
            if (job.valid) {
                result.valid_count++;
            } else {
                result.invalid_count++;
                if (!result.has_error) {
                    result.has_error = true;
                    result.first_error_tx = job.tx_index;
                    result.first_error_input = job.input_index;
                    result.first_error_code = job.error;
                }
            }
        } else {
            result.skipped_count++;
        }
    }

    GPU_LOG_INFO("Batch validation complete: %u valid, %u invalid, %u skipped (GPU: %.2fms, setup: %.2fms)",
                 result.valid_count, result.invalid_count, result.skipped_count,
                 result.gpu_time_ms, result.setup_time_ms);

    return result;
}

bool GPUBatchValidator::GetJobResult(size_t job_index, GPUScriptError& error) const
{
    if (job_index >= m_job_count) {
        return false;
    }
    error = m_jobs[job_index].error;
    return m_jobs[job_index].valid;
}

const ScriptValidationJob* GPUBatchValidator::GetJob(size_t job_index) const
{
    if (job_index >= m_job_count) {
        return nullptr;
    }
    return &m_jobs[job_index];
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// ============================================================================
// Helper: Parse witness stack into individual items
// ============================================================================
__device__ inline bool ParseWitnessStack(
    const uint8_t* witness_data,
    uint32_t witness_size,
    uint32_t witness_count,
    GPUScriptContext* ctx)
{
    if (!witness_data || witness_size == 0 || witness_count == 0) return true;

    const uint8_t* ptr = witness_data;
    uint32_t remaining = witness_size;

    for (uint32_t i = 0; i < witness_count && remaining > 0; i++) {
        // Read item length (simplified: 1 byte length for items < 253 bytes)
        uint32_t item_len = ptr[0];
        ptr++; remaining--;

        if (item_len > remaining) return false;

        // Push to stack
        if (!stack_push(ctx, ptr, item_len)) return false;

        ptr += item_len;
        remaining -= item_len;
    }

    return true;
}

// ============================================================================
// Fast-path validation for simple scripts using local memory
// Returns true if handled (success or failure), false if needs full interpreter
// ============================================================================
__device__ inline bool ValidateSimpleScript(
    ScriptValidationJob& job,
    const uint8_t* scriptpubkey,
    const uint8_t* scriptsig,
    const uint8_t* witness,
    ScriptType script_type)
{
    // Use small local context - fits in local memory with L1/L2 caching
    GPUScriptContextSmall ctx;
    ctx.sigversion = job.sigversion;
    ctx.verify_flags = job.verify_flags;
    ctx.input_amount = job.amount;
    ctx.input_sequence = job.sequence;

    switch (script_type) {
        case SCRIPT_TYPE_P2WPKH:
        {
            // P2WPKH: witness = [signature, pubkey]
            // Verify: pubkey hash matches, ECDSA signature valid
            if (job.witness_count != 2) {
                job.error = GPU_SCRIPT_ERR_WITNESS_PROGRAM_MISMATCH;
                job.valid = false;
                return true;
            }
            // Safety check: witness_count > 0 but witness_total_size == 0 is inconsistent
            if (job.witness_total_size == 0) {
                return false;  // Fall back to CPU
            }

            // Parse witness: [sig_len, sig..., pubkey_len, pubkey...]
            const uint8_t* ptr = witness;
            uint8_t sig_len = ptr[0];
            if (sig_len < 9 || sig_len > 73) {
                job.error = GPU_SCRIPT_ERR_SIG_DER;
                job.valid = false;
                return true;
            }
            const uint8_t* sig = ptr + 1;
            ptr += 1 + sig_len;
            uint8_t pubkey_len = ptr[0];
            const uint8_t* pubkey = ptr + 1;

            // Verify pubkey length (33 compressed or 65 uncompressed)
            if (pubkey_len != 33 && pubkey_len != 65) {
                job.error = GPU_SCRIPT_ERR_WITNESS_PUBKEYTYPE;
                job.valid = false;
                return true;
            }

            // Hash pubkey and compare to witness program
            uint8_t pubkey_hash[20];
            hash160(pubkey, pubkey_len, pubkey_hash);

            // scriptpubkey[2..22] is the 20-byte hash
            bool hash_match = true;
            for (int i = 0; i < 20; i++) {
                if (pubkey_hash[i] != scriptpubkey[2 + i]) {
                    hash_match = false;
                    break;
                }
            }
            if (!hash_match) {
                job.error = GPU_SCRIPT_ERR_WITNESS_PROGRAM_MISMATCH;
                job.valid = false;
                return true;
            }

            // Extract actual DER sig length (signature minus hashtype byte at end)
            uint32_t actual_sig_len = sig_len - 1;  // DER sig without hashtype

            // Check if sighash is precomputed
            if (!job.sighash_valid) {
                // Cannot verify without sighash - fall back to full interpreter
                return false;
            }

            // Low-S check for BIP146
            if (job.verify_flags & GPU_SCRIPT_VERIFY_LOW_S) {
                secp256k1::Scalar r, s;
                if (!secp256k1::sig_parse_der_simple(r, s, sig, actual_sig_len)) {
                    job.error = GPU_SCRIPT_ERR_SIG_DER;
                    job.valid = false;
                    return true;
                }
                if (!secp256k1::sig_has_low_s(s)) {
                    job.error = GPU_SCRIPT_ERR_SIG_HIGH_S;
                    job.valid = false;
                    return true;
                }
            }

            // ECDSA signature verification
            bool sig_valid = secp256k1::ecdsa_verify(
                sig, actual_sig_len,
                job.sighash.data,
                pubkey, pubkey_len
            );

            if (!sig_valid) {
                job.error = GPU_SCRIPT_ERR_SIG_ECDSA;
                job.valid = false;
                return true;
            }

            job.valid = true;
            job.error = GPU_SCRIPT_ERR_OK;
            return true;
        }

        case SCRIPT_TYPE_P2TR:
        {
            // P2TR key-path: witness = [signature] (64 or 65 bytes)
            // P2TR script-path: witness = [..., script, control_block]
            if (job.witness_count == 1) {
                // Key-path spend
                // Safety check: witness_count > 0 but witness_total_size == 0 is inconsistent
                if (job.witness_total_size == 0) {
                    // Data inconsistency - fall back to CPU for safety
                    return false;
                }
                uint8_t sig_len = witness[0];
                if (sig_len != 64 && sig_len != 65) {
                    printf("[GPU DEBUG] SCHNORR_SIG_SIZE error: expected 64 or 65, got %u\n", sig_len);
                    job.error = GPU_SCRIPT_ERR_SCHNORR_SIG_SIZE;
                    job.valid = false;
                    return true;
                }
                const uint8_t* sig = witness + 1;

                // Get sighash type
                uint8_t hashtype = sighash::SIGHASH_DEFAULT;
                if (sig_len == 65) {
                    hashtype = sig[64];
                    // Validate hashtype
                    if (hashtype == 0x00) {
                        // SIGHASH_DEFAULT - valid
                    } else {
                        uint8_t baseType = hashtype & 0x03;
                        if (baseType < 0x01 || baseType > 0x03) {
                            job.error = GPU_SCRIPT_ERR_SIG_HASHTYPE;
                            job.valid = false;
                            return true;
                        }
                    }
                }

                // Check if sighash is precomputed
                if (!job.sighash_valid) {
                    // Cannot verify without sighash - fall back to full interpreter
                    return false;
                }

                // Get x-only pubkey from scriptPubKey (bytes 2-33)
                const uint8_t* pubkey = scriptpubkey + 2;

                // Schnorr signature verification (BIP340)
                bool sig_valid = secp256k1::schnorr_verify(
                    sig,
                    job.sighash.data,
                    32,
                    pubkey
                );

                if (!sig_valid) {
                    job.error = GPU_SCRIPT_ERR_SCHNORR_SIG;
                    job.valid = false;
                    return true;
                }

                job.valid = true;
                job.error = GPU_SCRIPT_ERR_OK;
                return true;
            }
            // Script-path needs full interpreter
            return false;
        }

        case SCRIPT_TYPE_P2PKH:
        {
            // P2PKH: scriptsig = <sig> <pubkey>
            // scriptpubkey: OP_DUP OP_HASH160 <20-byte-hash> OP_EQUALVERIFY OP_CHECKSIG
            // Max stack depth = 4, perfect for fast path

            // Parse scriptsig to get sig and pubkey
            if (job.scriptsig_size < 2) {
                job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                job.valid = false;
                return true;
            }

            const uint8_t* ptr = scriptsig;
            uint8_t sig_len = ptr[0];
            if (sig_len < 9 || sig_len > 73 || 1 + sig_len >= job.scriptsig_size) {
                job.error = GPU_SCRIPT_ERR_SIG_DER;
                job.valid = false;
                return true;
            }
            const uint8_t* sig = ptr + 1;
            ptr += 1 + sig_len;

            uint8_t pubkey_len = ptr[0];
            if (pubkey_len != 33 && pubkey_len != 65) {
                job.error = GPU_SCRIPT_ERR_PUBKEYTYPE;
                job.valid = false;
                return true;
            }
            const uint8_t* pubkey = ptr + 1;

            // Hash pubkey and compare to expected hash in scriptpubkey
            // scriptpubkey: 76 a9 14 <20 bytes> 88 ac
            uint8_t pubkey_hash[20];
            hash160(pubkey, pubkey_len, pubkey_hash);

            // Compare with scriptpubkey[3..23]
            bool hash_match = true;
            for (int i = 0; i < 20; i++) {
                if (pubkey_hash[i] != scriptpubkey[3 + i]) {
                    hash_match = false;
                    break;
                }
            }
            if (!hash_match) {
                job.error = GPU_SCRIPT_ERR_EQUALVERIFY;
                job.valid = false;
                return true;
            }

            // Check if sighash is precomputed
            if (!job.sighash_valid) {
                // Cannot verify without sighash - fall back to full interpreter
                return false;
            }

            // Get actual DER sig length (signature minus hashtype byte at end)
            uint32_t actual_sig_len = sig_len - 1;

            // Low-S check for BIP146
            if (job.verify_flags & GPU_SCRIPT_VERIFY_LOW_S) {
                secp256k1::Scalar r, s;
                if (!secp256k1::sig_parse_der_simple(r, s, sig, actual_sig_len)) {
                    job.error = GPU_SCRIPT_ERR_SIG_DER;
                    job.valid = false;
                    return true;
                }
                if (!secp256k1::sig_has_low_s(s)) {
                    job.error = GPU_SCRIPT_ERR_SIG_HIGH_S;
                    job.valid = false;
                    return true;
                }
            }

            // ECDSA signature verification
            bool sig_valid = secp256k1::ecdsa_verify(
                sig, actual_sig_len,
                job.sighash.data,
                pubkey, pubkey_len
            );

            if (!sig_valid) {
                job.error = GPU_SCRIPT_ERR_SIG_ECDSA;
                job.valid = false;
                return true;
            }

            job.valid = true;
            job.error = GPU_SCRIPT_ERR_OK;
            return true;
        }

        case SCRIPT_TYPE_P2PK:
        {
            // P2PK: scriptsig = <sig>
            // scriptpubkey: <pubkey> OP_CHECKSIG
            // Max stack depth = 2, trivial fast path

            if (job.scriptsig_size < 2) {
                job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                job.valid = false;
                return true;
            }

            uint8_t sig_len = scriptsig[0];
            if (sig_len < 9 || sig_len > 73 || 1 + sig_len > job.scriptsig_size) {
                job.error = GPU_SCRIPT_ERR_SIG_DER;
                job.valid = false;
                return true;
            }
            const uint8_t* sig = scriptsig + 1;

            // Get pubkey from scriptpubkey (first byte is push length)
            uint8_t pubkey_len = scriptpubkey[0];
            if (pubkey_len != 33 && pubkey_len != 65) {
                job.error = GPU_SCRIPT_ERR_PUBKEYTYPE;
                job.valid = false;
                return true;
            }
            const uint8_t* pubkey = scriptpubkey + 1;

            // Check if sighash is precomputed
            if (!job.sighash_valid) {
                // Cannot verify without sighash - fall back to full interpreter
                return false;
            }

            // Get actual DER sig length (minus hashtype byte at end)
            uint32_t actual_sig_len = sig_len - 1;

            // Low-S check for BIP146
            if (job.verify_flags & GPU_SCRIPT_VERIFY_LOW_S) {
                secp256k1::Scalar r, s;
                if (!secp256k1::sig_parse_der_simple(r, s, sig, actual_sig_len)) {
                    job.error = GPU_SCRIPT_ERR_SIG_DER;
                    job.valid = false;
                    return true;
                }
                if (!secp256k1::sig_has_low_s(s)) {
                    job.error = GPU_SCRIPT_ERR_SIG_HIGH_S;
                    job.valid = false;
                    return true;
                }
            }

            // ECDSA signature verification
            bool sig_valid = secp256k1::ecdsa_verify(
                sig, actual_sig_len,
                job.sighash.data,
                pubkey, pubkey_len
            );

            if (!sig_valid) {
                job.error = GPU_SCRIPT_ERR_SIG_ECDSA;
                job.valid = false;
                return true;
            }

            job.valid = true;
            job.error = GPU_SCRIPT_ERR_OK;
            return true;
        }

        case SCRIPT_TYPE_P2SH:
        {
            // P2SH-P2WPKH fast-path: scriptsig = <0014{20-byte-hash}>, witness = [sig, pubkey]
            // Check for P2SH-wrapped P2WPKH (most common P2SH use case)
            if (job.scriptsig_size == 23 &&
                scriptsig[0] == 0x16 &&  // Push 22 bytes
                scriptsig[1] == 0x00 &&  // OP_0 (witness version)
                scriptsig[2] == 0x14 &&  // Push 20 bytes
                job.witness_count == 2)
            {
                // This is P2SH-P2WPKH
                // Verify the hash of the redeem script matches P2SH
                uint8_t redeem_script[22];
                for (int i = 0; i < 22; i++) {
                    redeem_script[i] = scriptsig[1 + i];
                }

                uint8_t redeem_hash[20];
                hash160(redeem_script, 22, redeem_hash);

                // Compare with scriptPubKey hash (bytes 2-21 of P2SH: a9 14 <hash> 87)
                bool hash_match = true;
                for (int i = 0; i < 20; i++) {
                    if (redeem_hash[i] != scriptpubkey[2 + i]) {
                        hash_match = false;
                        break;
                    }
                }
                if (!hash_match) {
                    job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                    job.valid = false;
                    return true;
                }

                // Now validate the witness for P2WPKH
                const uint8_t* ptr = witness;
                uint8_t sig_len = ptr[0];
                if (sig_len < 9 || sig_len > 73) {
                    job.error = GPU_SCRIPT_ERR_SIG_DER;
                    job.valid = false;
                    return true;
                }
                const uint8_t* sig = ptr + 1;
                ptr += 1 + sig_len;
                uint8_t pubkey_len = ptr[0];
                const uint8_t* pubkey = ptr + 1;

                if (pubkey_len != 33 && pubkey_len != 65) {
                    job.error = GPU_SCRIPT_ERR_WITNESS_PUBKEYTYPE;
                    job.valid = false;
                    return true;
                }

                // Hash pubkey and compare to witness program in redeem script
                uint8_t pubkey_hash[20];
                hash160(pubkey, pubkey_len, pubkey_hash);

                // redeem_script[2..22] is the 20-byte hash
                hash_match = true;
                for (int i = 0; i < 20; i++) {
                    if (pubkey_hash[i] != redeem_script[2 + i]) {
                        hash_match = false;
                        break;
                    }
                }
                if (!hash_match) {
                    job.error = GPU_SCRIPT_ERR_WITNESS_PROGRAM_MISMATCH;
                    job.valid = false;
                    return true;
                }

                // Check if sighash is precomputed
                if (!job.sighash_valid) {
                    return false;  // Fall back to full interpreter
                }

                // Get actual DER sig length (minus hashtype byte at end)
                uint32_t actual_sig_len = sig_len - 1;

                // Low-S check
                if (job.verify_flags & GPU_SCRIPT_VERIFY_LOW_S) {
                    secp256k1::Scalar r, s;
                    if (!secp256k1::sig_parse_der_simple(r, s, sig, actual_sig_len)) {
                        job.error = GPU_SCRIPT_ERR_SIG_DER;
                        job.valid = false;
                        return true;
                    }
                    if (!secp256k1::sig_has_low_s(s)) {
                        job.error = GPU_SCRIPT_ERR_SIG_HIGH_S;
                        job.valid = false;
                        return true;
                    }
                }

                // ECDSA verification
                bool sig_valid = secp256k1::ecdsa_verify(
                    sig, actual_sig_len,
                    job.sighash.data,
                    pubkey, pubkey_len
                );

                if (!sig_valid) {
                    job.error = GPU_SCRIPT_ERR_SIG_ECDSA;
                    job.valid = false;
                    return true;
                }

                job.valid = true;
                job.error = GPU_SCRIPT_ERR_OK;
                return true;
            }
            // Other P2SH types need full interpreter
            return false;
        }

        case SCRIPT_TYPE_NULL_DATA:
        {
            // OP_RETURN is always unspendable
            job.error = GPU_SCRIPT_ERR_OP_RETURN;
            job.valid = false;
            return true;
        }

        case SCRIPT_TYPE_WITNESS_UNKNOWN:
        {
            // Future witness versions
            if (job.verify_flags & GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_WITNESS) {
                job.error = GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM;
                job.valid = false;
            } else {
                // Anyone-can-spend for future versions
                job.valid = true;
                job.error = GPU_SCRIPT_ERR_OK;
            }
            return true;
        }

        default:
            return false;  // Need full interpreter
    }
}

// ============================================================================
// Full Script Validation Kernel - Supports ALL Script Types
// Uses hybrid memory: local for simple scripts, global for complex
// ============================================================================
__global__ void BatchValidateScriptsKernel(
    ScriptValidationJob* jobs,
    const uint8_t* scriptpubkey_blob,
    const uint8_t* scriptsig_blob,
    const uint8_t* witness_blob,
    GPUScriptContext* contexts,
    uint32_t job_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= job_count) return;

    ScriptValidationJob& job = jobs[idx];
    job.validated = true;

    // Get script pointers
    const uint8_t* scriptpubkey = scriptpubkey_blob + job.scriptpubkey_offset;
    const uint8_t* scriptsig = scriptsig_blob + job.scriptsig_offset;
    const uint8_t* witness = witness_blob + job.witness_offset;

    // Determine script type
    ScriptType script_type = IdentifyScriptTypeDevice(scriptpubkey, job.scriptpubkey_size);

    // Try fast-path with local memory first
    if (ValidateSimpleScript(job, scriptpubkey, scriptsig, witness, script_type)) {
        return;  // Handled by fast path
    }

    // Complex script - use global memory context
    GPUScriptContext& ctx = contexts[idx];

    // Reset context for this job
    ctx.stack_size = 0;
    ctx.altstack_size = 0;
    ctx.conditions.size = 0;
    ctx.conditions.first_false_pos = GPUConditionStack::NO_FALSE;
    ctx.pc = 0;
    ctx.opcode_count = 0;
    ctx.error = GPU_SCRIPT_ERR_OK;
    ctx.success = false;

    ctx.sigversion = job.sigversion;
    ctx.verify_flags = job.verify_flags;
    ctx.input_amount = job.amount;
    ctx.input_sequence = job.sequence;

    // Copy precomputed sighash from job to context
    ctx.precomputed_sighash_valid = job.sighash_valid;
    if (job.sighash_valid) {
        for (int i = 0; i < 32; i++) {
            ctx.precomputed_sighash.data[i] = job.sighash.data[i];
        }
    }

    bool success = false;
    bool has_witness = (job.witness_count > 0 && job.witness_total_size > 0);

    // ==========================================================================
    // STEP 1: Execute scriptSig (for non-SegWit) or verify it's empty (for native SegWit)
    // ==========================================================================

    if (script_type == SCRIPT_TYPE_P2WPKH || script_type == SCRIPT_TYPE_P2WSH ||
        script_type == SCRIPT_TYPE_P2TR || script_type == SCRIPT_TYPE_WITNESS_UNKNOWN) {
        // Native SegWit: scriptSig must be empty
        if (job.scriptsig_size > 0) {
            // Check for P2SH-wrapped SegWit
            // For now, treat non-empty scriptSig as the witness program push
            // This needs proper handling for P2SH-P2WPKH and P2SH-P2WSH
        }
    } else {
        // Non-SegWit or P2SH: Execute scriptSig to push data onto stack
        if (job.scriptsig_size > 0) {
            if (!EvalScript(&ctx, scriptsig, job.scriptsig_size)) {
                job.error = ctx.error;
                job.valid = false;
                return;
            }
        }
    }

    // ==========================================================================
    // STEP 2: Handle based on script type
    // ==========================================================================

    switch (script_type) {
        case SCRIPT_TYPE_P2PKH:
        case SCRIPT_TYPE_P2PK:
        case SCRIPT_TYPE_MULTISIG:
        case SCRIPT_TYPE_NONSTANDARD:
        {
            // Legacy scripts: Execute scriptPubKey with stack from scriptSig
            // P2PK: <sig> | <pubkey> OP_CHECKSIG
            // MULTISIG: <dummy> <sig1>...<sigN> | OP_M <pub1>...<pubN> OP_N OP_CHECKMULTISIG
            // NONSTANDARD: Any other valid script (executes as-is)
            if (job.scriptpubkey_size > 0) {
                if (!EvalScript(&ctx, scriptpubkey, job.scriptpubkey_size)) {
                    job.error = ctx.error;
                    job.valid = false;
                    return;
                }
            }

            // Check final stack state
            if (ctx.stack_size == 0) {
                job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                job.valid = false;
                return;
            }

            success = CastToBool(stacktop(&ctx, -1));
            break;
        }

        case SCRIPT_TYPE_P2SH:
        {
            // P2SH: scriptSig ends with serialized redeemScript
            // First execute scriptPubKey (OP_HASH160 <hash> OP_EQUAL)
            if (job.scriptpubkey_size > 0) {
                // Save the top of stack (serialized redeemScript) before executing scriptPubKey
                if (ctx.stack_size == 0) {
                    job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                    job.valid = false;
                    return;
                }

                GPUStackElement redeem_serialized = stacktop(&ctx, -1);

                if (!EvalScript(&ctx, scriptpubkey, job.scriptpubkey_size)) {
                    job.error = ctx.error;
                    job.valid = false;
                    return;
                }

                // scriptPubKey should leave true on stack
                if (ctx.stack_size == 0 || !CastToBool(stacktop(&ctx, -1))) {
                    job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                    job.valid = false;
                    return;
                }

                // Now deserialize and execute the redeemScript
                // Clear stack and re-execute scriptSig (minus the redeemScript push)
                // then execute the redeemScript

                // Reset context for redeemScript execution
                ctx.stack_size = 0;
                ctx.altstack_size = 0;

                // Re-execute scriptSig
                if (job.scriptsig_size > 0) {
                    if (!EvalScript(&ctx, scriptsig, job.scriptsig_size)) {
                        job.error = ctx.error;
                        job.valid = false;
                        return;
                    }
                }

                // Pop the serialized redeemScript from stack and execute it
                if (ctx.stack_size > 0) {
                    GPUStackElement redeemScript;
                    stack_pop_to(&ctx, redeemScript);
                    if (!EvalScript(&ctx, redeemScript.data, redeemScript.size)) {
                        job.error = ctx.error;
                        job.valid = false;
                        return;
                    }
                }

                // Check if P2SH-wrapped SegWit (P2SH-P2WPKH or P2SH-P2WSH)
                if (has_witness && redeem_serialized.size >= 2) {
                    if (redeem_serialized.data[0] == 0x00 && redeem_serialized.data[1] == 0x14) {
                        // P2SH-P2WPKH: Execute witness program
                        ctx.sigversion = GPU_SIGVERSION_WITNESS_V0;
                        ctx.stack_size = 0;

                        if (!ParseWitnessStack(witness, job.witness_total_size, job.witness_count, &ctx)) {
                            job.error = GPU_SCRIPT_ERR_WITNESS_MALLEATED;
                            job.valid = false;
                            return;
                        }

                        // Construct implied P2PKH script
                        uint8_t implied_script[25];
                        implied_script[0] = GPU_OP_DUP;
                        implied_script[1] = GPU_OP_HASH160;
                        implied_script[2] = 0x14;
                        for (int i = 0; i < 20; i++) {
                            implied_script[3 + i] = redeem_serialized.data[2 + i];
                        }
                        implied_script[23] = GPU_OP_EQUALVERIFY;
                        implied_script[24] = GPU_OP_CHECKSIG;

                        if (!EvalScript(&ctx, implied_script, 25)) {
                            job.error = ctx.error;
                            job.valid = false;
                            return;
                        }
                    } else if (redeem_serialized.data[0] == 0x00 && redeem_serialized.data[1] == 0x20) {
                        // P2SH-P2WSH: Execute witness script
                        ctx.sigversion = GPU_SIGVERSION_WITNESS_V0;
                        ctx.stack_size = 0;

                        if (!ParseWitnessStack(witness, job.witness_total_size, job.witness_count, &ctx)) {
                            job.error = GPU_SCRIPT_ERR_WITNESS_MALLEATED;
                            job.valid = false;
                            return;
                        }

                        // Last witness item is the witnessScript
                        if (ctx.stack_size > 0) {
                            GPUStackElement witnessScript;
                            stack_pop_to(&ctx, witnessScript);
                            if (!EvalScript(&ctx, witnessScript.data, witnessScript.size)) {
                                job.error = ctx.error;
                                job.valid = false;
                                return;
                            }
                        }
                    }
                }
            }

            if (ctx.stack_size == 0) {
                job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                job.valid = false;
                return;
            }
            success = CastToBool(stacktop(&ctx, -1));
            break;
        }

        case SCRIPT_TYPE_P2WPKH:
        {
            // Native P2WPKH: witness = [<sig>, <pubkey>]
            ctx.sigversion = GPU_SIGVERSION_WITNESS_V0;

            // Parse witness stack onto execution stack
            if (!ParseWitnessStack(witness, job.witness_total_size, job.witness_count, &ctx)) {
                job.error = GPU_SCRIPT_ERR_WITNESS_MALLEATED;
                job.valid = false;
                return;
            }

            // Construct implied P2PKH script from pubkey hash in scriptPubKey
            uint8_t implied_script[25];
            implied_script[0] = GPU_OP_DUP;
            implied_script[1] = GPU_OP_HASH160;
            implied_script[2] = 0x14;
            if (job.scriptpubkey_size >= 22) {
                for (int i = 0; i < 20; i++) {
                    implied_script[3 + i] = scriptpubkey[2 + i];
                }
            }
            implied_script[23] = GPU_OP_EQUALVERIFY;
            implied_script[24] = GPU_OP_CHECKSIG;

            if (!EvalScript(&ctx, implied_script, 25)) {
                job.error = ctx.error;
                job.valid = false;
                return;
            }

            if (ctx.stack_size == 0) {
                job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                job.valid = false;
                return;
            }
            success = CastToBool(stacktop(&ctx, -1));
            break;
        }

        case SCRIPT_TYPE_P2WSH:
        {
            // Native P2WSH: witness = [<item1>, <item2>, ..., <witnessScript>]
            ctx.sigversion = GPU_SIGVERSION_WITNESS_V0;

            // Parse witness stack
            if (!ParseWitnessStack(witness, job.witness_total_size, job.witness_count, &ctx)) {
                job.error = GPU_SCRIPT_ERR_WITNESS_MALLEATED;
                job.valid = false;
                return;
            }

            // Last witness item is the witnessScript - pop and execute it
            if (ctx.stack_size > 0) {
                GPUStackElement witnessScript;
                stack_pop_to(&ctx, witnessScript);

                // Verify witnessScript hash matches scriptPubKey
                // SHA256(witnessScript) should equal bytes 2-33 of scriptPubKey
                uint8_t script_hash[32];
                sha256(witnessScript.data, witnessScript.size, script_hash);

                bool hash_match = true;
                if (job.scriptpubkey_size >= 34) {
                    for (int i = 0; i < 32 && hash_match; i++) {
                        if (script_hash[i] != scriptpubkey[2 + i]) {
                            hash_match = false;
                        }
                    }
                } else {
                    hash_match = false;
                }

                if (!hash_match) {
                    job.error = GPU_SCRIPT_ERR_WITNESS_PROGRAM_MISMATCH;
                    job.valid = false;
                    return;
                }

                // Execute the witnessScript
                if (!EvalScript(&ctx, witnessScript.data, witnessScript.size)) {
                    job.error = ctx.error;
                    job.valid = false;
                    return;
                }
            }

            if (ctx.stack_size == 0) {
                job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                job.valid = false;
                return;
            }
            success = CastToBool(stacktop(&ctx, -1));
            break;
        }

        case SCRIPT_TYPE_P2TR:
        {
            // Taproot: Check if key-path or script-path
            ctx.sigversion = GPU_SIGVERSION_TAPROOT;

            if (job.witness_count == 1) {
                // Key-path spend: witness = [<signature>]
                const uint8_t* wptr = witness;
                uint32_t sig_len = wptr[0];
                wptr++;

                // Schnorr signature must be 64 bytes (no sighash type) or 65 bytes (with sighash type)
                if (sig_len != 64 && sig_len != 65) {
                    job.error = GPU_SCRIPT_ERR_SCHNORR_SIG_SIZE;
                    job.valid = false;
                    return;
                }

                const uint8_t* sig = wptr;
                const uint8_t* pubkey = scriptpubkey + 2;  // x-only pubkey from P2TR output

                // Get sighash type from signature or default to SIGHASH_DEFAULT (0x00)
                uint8_t hashtype = sighash::SIGHASH_DEFAULT;
                if (sig_len == 65) {
                    hashtype = sig[64];
                    // SIGHASH_DEFAULT (0x00) is only valid for Taproot
                    // Other values: 0x01 (ALL), 0x02 (NONE), 0x03 (SINGLE)
                    // with optional 0x80 (ANYONECANPAY) flag
                    if (hashtype == 0x00) {
                        // SIGHASH_DEFAULT is valid, treated as SIGHASH_ALL
                    } else {
                        uint8_t baseType = hashtype & 0x03;
                        if (baseType < 0x01 || baseType > 0x03) {
                            job.error = GPU_SCRIPT_ERR_SIG_HASHTYPE;
                            job.valid = false;
                            return;
                        }
                    }
                }

                // Check if sighash was precomputed by caller
                // If not, we need the transaction data to compute it
                if (!job.sighash_valid) {
                    // Sighash must be precomputed by the caller for Taproot
                    // This requires full transaction context which should be
                    // passed via validation.cpp before batch execution
                    job.error = GPU_SCRIPT_ERR_UNKNOWN_ERROR;
                    job.valid = false;
                    return;
                }

                // Verify Schnorr signature against precomputed sighash
                success = secp256k1::schnorr_verify(sig, job.sighash.data, 32, pubkey);
                if (!success) {
                    job.error = GPU_SCRIPT_ERR_SCHNORR_SIG;
                }
            } else if (job.witness_count >= 2) {
                // Script-path spend: witness = [<stack items>..., <script>, <control block>]
                ctx.sigversion = GPU_SIGVERSION_TAPSCRIPT;

                // Parse witness stack
                if (!ParseWitnessStack(witness, job.witness_total_size, job.witness_count, &ctx)) {
                    job.error = GPU_SCRIPT_ERR_WITNESS_MALLEATED;
                    job.valid = false;
                    return;
                }

                // Pop control block (last item) and tapscript (second to last)
                if (ctx.stack_size < 2) {
                    job.error = GPU_SCRIPT_ERR_WITNESS_PROGRAM_WRONG_LENGTH;
                    job.valid = false;
                    return;
                }

                GPUStackElement control_block, tapscript;
                stack_pop_to(&ctx, control_block);
                stack_pop_to(&ctx, tapscript);

                // Validate control block minimum size (1 byte version + 32 byte internal key)
                if (control_block.size < 33) {
                    job.error = GPU_SCRIPT_ERR_TAPROOT_WRONG_CONTROL_SIZE;
                    job.valid = false;
                    return;
                }

                // Control block size must be 33 + 32*k for some k >= 0
                if ((control_block.size - 33) % 32 != 0) {
                    job.error = GPU_SCRIPT_ERR_TAPROOT_WRONG_CONTROL_SIZE;
                    job.valid = false;
                    return;
                }

                // Check merkle path depth doesn't exceed 128
                uint32_t path_len = (control_block.size - 33) / 32;
                if (path_len > 128) {
                    job.error = GPU_SCRIPT_ERR_TAPROOT_WRONG_CONTROL_SIZE;
                    job.valid = false;
                    return;
                }

                // Extract and validate leaf version (must be valid tapscript version)
                uint8_t leaf_version = control_block.data[0] & 0xFE;
                if (leaf_version != 0xC0) {  // TAPROOT_LEAF_TAPSCRIPT = 0xC0
                    // Unknown leaf version - we could allow it but for safety reject
                    job.error = GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION;
                    job.valid = false;
                    return;
                }

                // Get output pubkey from scriptPubKey (bytes 2-33 of P2TR output)
                const uint8_t* output_pubkey = scriptpubkey + 2;

                // Verify the merkle proof: control block commits to this script
                if (!secp256k1::verify_taproot_script_path(
                        output_pubkey,
                        control_block.data,
                        control_block.size,
                        tapscript.data,
                        tapscript.size)) {
                    job.error = GPU_SCRIPT_ERR_WITNESS_PROGRAM_MISMATCH;
                    job.valid = false;
                    return;
                }

                // Store tapleaf hash for sighash computation in CHECKSIG operations
                uint8_t tapleaf_hash[32];
                secp256k1::compute_tapleaf_hash(tapleaf_hash, leaf_version, tapscript.data, tapscript.size);
                for (int i = 0; i < 32; i++) {
                    ctx.execdata.tapleaf_hash.data[i] = tapleaf_hash[i];
                }
                ctx.execdata.tapleaf_hash_init = true;

                // Execute the tapscript
                if (!EvalScript(&ctx, tapscript.data, tapscript.size)) {
                    job.error = ctx.error;
                    job.valid = false;
                    return;
                }

                if (ctx.stack_size == 0) {
                    job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                    job.valid = false;
                    return;
                }
                success = CastToBool(stacktop(&ctx, -1));
            } else {
                job.error = GPU_SCRIPT_ERR_WITNESS_PROGRAM_WITNESS_EMPTY;
                job.valid = false;
                return;
            }
            break;
        }

        case SCRIPT_TYPE_NULL_DATA:
        {
            // OP_RETURN scripts are provably unspendable
            // They should never succeed validation - this is by design
            job.error = GPU_SCRIPT_ERR_OP_RETURN;
            job.valid = false;
            return;
        }

        case SCRIPT_TYPE_WITNESS_UNKNOWN:
        {
            // Future witness versions (OP_2 through OP_16 with 2-40 byte data)
            // BIP141: "If the version byte is 2 to 16, no further interpretation
            // of the witness program or witness stack happens, and there is no
            // size restriction for any push opcode. These versions are reserved
            // for future upgrades."
            //
            // With DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM flag: reject
            // Without the flag: treat as anyone-can-spend (success)
            if (ctx.verify_flags & GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_WITNESS) {
                job.error = GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM;
                job.valid = false;
                return;
            }
            // Otherwise, success-by-default for future witness versions
            success = true;
            break;
        }

        default:
        {
            // Unknown script type - try to execute as legacy script anyway
            // This handles any valid script that doesn't match standard patterns
            if (job.scriptpubkey_size > 0) {
                if (!EvalScript(&ctx, scriptpubkey, job.scriptpubkey_size)) {
                    job.error = ctx.error;
                    job.valid = false;
                    return;
                }
            }

            if (ctx.stack_size == 0) {
                job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
                job.valid = false;
                return;
            }
            success = CastToBool(stacktop(&ctx, -1));
            break;
        }
    }

    // ==========================================================================
    // STEP 3: Final validation
    // ==========================================================================

    // Check CLEANSTACK if required
    if ((ctx.verify_flags & GPU_SCRIPT_VERIFY_CLEANSTACK) != 0) {
        if (ctx.stack_size != 1) {
            job.error = GPU_SCRIPT_ERR_CLEANSTACK;
            job.valid = false;
            return;
        }
    }

    job.valid = success;
    if (!success && job.error == GPU_SCRIPT_ERR_OK) {
        job.error = GPU_SCRIPT_ERR_EVAL_FALSE;
    }
}

// NOTE: Sighash precomputation is currently done on CPU before queueing jobs
// because sighash computation requires full transaction context (all inputs,
// outputs, prevouts, amounts) which would require significant data transfer.
//
// For maximum GPU utilization in future, consider:
// 1. Passing serialized transaction data to GPU
// 2. Using this kernel to compute sighashes in parallel
// 3. Then running BatchValidateScriptsKernel
//
// Current design: validation.cpp computes sighashes and passes via job.sighash
__global__ void BatchComputeSigHashKernel(
    ScriptValidationJob* jobs,
    const uint8_t* tx_data,
    uint32_t job_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= job_count) return;

    ScriptValidationJob& job = jobs[idx];

    // Skip if sighash already computed by CPU
    if (job.sighash_valid) return;

    // If tx_data is provided, compute sighash on GPU
    // Currently tx_data is not populated, so sighash must be precomputed
    if (tx_data == nullptr) {
        job.sighash_valid = false;
        return;
    }

    // Future: Parse tx_data and compute sighash using gpu_sighash.cuh
    // For now, sighash computation is done on CPU before batch execution
    job.sighash_valid = false;
}

__global__ void BatchVerifyECDSAKernel(
    const uint8_t* sighashes,
    const uint8_t* signatures,
    const uint32_t* sig_sizes,
    const uint8_t* pubkeys,
    const uint32_t* pubkey_sizes,
    bool* results,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Get pointers for this verification
    const uint8_t* sighash = sighashes + (idx * 32);
    uint32_t sig_offset = 0;
    uint32_t pubkey_offset = 0;

    // Calculate offsets (simplified - assumes fixed max sizes)
    for (uint32_t i = 0; i < idx; i++) {
        sig_offset += sig_sizes[i];
        pubkey_offset += pubkey_sizes[i];
    }

    const uint8_t* sig = signatures + sig_offset;
    uint32_t sig_size = sig_sizes[idx];
    const uint8_t* pubkey = pubkeys + pubkey_offset;
    uint32_t pubkey_size = pubkey_sizes[idx];

    // Parse signature (DER format)
    secp256k1::Scalar r, s;
    if (!secp256k1::sig_parse_der_simple(r, s, sig, sig_size)) {
        results[idx] = false;
        return;
    }

    // Parse public key
    secp256k1::AffinePoint pk;
    if (!secp256k1::pubkey_parse(pk, pubkey, pubkey_size)) {
        results[idx] = false;
        return;
    }

    // Verify
    results[idx] = secp256k1::ecdsa_verify_core(sighash, r, s, pk);
}

__global__ void BatchVerifySchnorrKernel(
    const uint8_t* sighashes,
    const uint8_t* signatures,
    const uint8_t* pubkeys,
    bool* results,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* sighash = sighashes + (idx * 32);
    const uint8_t* sig = signatures + (idx * 64);
    const uint8_t* pubkey = pubkeys + (idx * 32);

    results[idx] = secp256k1::schnorr_verify(sig, sighash, 32, pubkey);
}

} // namespace gpu
