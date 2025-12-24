// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_BATCH_VALIDATOR_H
#define BITCOIN_GPU_KERNEL_GPU_BATCH_VALIDATOR_H

#include "gpu_types.h"
#include "gpu_utxo.h"
#include "gpu_script_types.cuh"

#include <cstdint>
#include <vector>
#include <memory>

// Forward declarations for Bitcoin Core types
class CTransaction;
class CScript;
class CTxOut;
class COutPoint;
class uint256;

namespace gpu {

// ============================================================================
// Script Validation Job - represents a single input to validate
// ============================================================================

struct ScriptValidationJob {
    // Transaction identification
    uint32_t tx_index;        // Index in block
    uint32_t input_index;     // Input index within transaction

    // Script data pointers (into GPU memory)
    uint32_t scriptpubkey_offset;   // Offset in scriptPubKey blob
    uint16_t scriptpubkey_size;     // Size of scriptPubKey
    uint32_t scriptsig_offset;      // Offset in scriptSig blob
    uint16_t scriptsig_size;        // Size of scriptSig

    // Witness data
    uint32_t witness_offset;        // Offset in witness blob
    uint16_t witness_count;         // Number of witness stack items
    uint16_t witness_total_size;    // Total witness data size

    // Transaction context
    int64_t amount;                 // Amount being spent (satoshis)
    uint32_t sequence;              // Input sequence number

    // Precomputed sighash components
    uint256_gpu sighash;            // Precomputed sighash (when available)
    bool sighash_valid;             // Whether sighash is precomputed

    // Validation flags and context
    uint32_t verify_flags;          // SCRIPT_VERIFY_* flags
    GPUSigVersion sigversion;       // Signature version

    // Result
    GPUScriptError error;           // Error code if validation failed
    bool validated;                 // Whether validation was attempted
    bool valid;                     // Whether validation passed
};

// ============================================================================
// Batch Validation Result
// ============================================================================

struct BatchValidationResult {
    uint32_t total_jobs;            // Total number of jobs queued
    uint32_t validated_count;       // Number of jobs validated
    uint32_t valid_count;           // Number of valid results
    uint32_t invalid_count;         // Number of invalid results
    uint32_t skipped_count;         // Number skipped (CPU fallback needed)

    // First error information
    bool has_error;
    uint32_t first_error_tx;        // Transaction index of first error
    uint32_t first_error_input;     // Input index of first error
    GPUScriptError first_error_code;// Error code

    // Performance metrics
    double gpu_time_ms;             // Time spent on GPU validation
    double setup_time_ms;           // Time spent preparing data
};

// ============================================================================
// GPU Batch Validator Class
// ============================================================================

class GPUBatchValidator {
public:
    // Configuration
    static constexpr size_t DEFAULT_MAX_JOBS = 100000;
    static constexpr size_t DEFAULT_SCRIPT_BLOB_SIZE = 64 * 1024 * 1024;  // 64MB
    static constexpr size_t DEFAULT_WITNESS_BLOB_SIZE = 128 * 1024 * 1024; // 128MB

    GPUBatchValidator();
    ~GPUBatchValidator();

    // Initialization
    bool Initialize(size_t max_jobs = DEFAULT_MAX_JOBS,
                   size_t script_blob_size = DEFAULT_SCRIPT_BLOB_SIZE,
                   size_t witness_blob_size = DEFAULT_WITNESS_BLOB_SIZE);
    bool IsInitialized() const { return m_initialized; }
    void Shutdown();

    // Batch preparation
    void BeginBatch();
    void EndBatch();

    // Queue a script validation job
    // Returns job index, or -1 if queue is full
    int QueueJob(
        uint32_t tx_index,
        uint32_t input_index,
        const uint8_t* scriptpubkey, uint32_t scriptpubkey_len,
        const uint8_t* scriptsig, uint32_t scriptsig_len,
        const uint8_t* witness, uint32_t witness_len, uint32_t witness_count,
        int64_t amount,
        uint32_t sequence,
        uint32_t verify_flags,
        GPUSigVersion sigversion
    );

    // Queue from Bitcoin Core types (convenience wrapper)
    int QueueTransaction(
        uint32_t tx_index,
        const CTransaction& tx,
        const std::vector<CTxOut>& spent_outputs,
        uint32_t verify_flags
    );

    // Execute validation
    BatchValidationResult ValidateBatch();

    // Get individual job results
    bool GetJobResult(size_t job_index, GPUScriptError& error) const;
    const ScriptValidationJob* GetJob(size_t job_index) const;

    // Statistics
    size_t GetQueuedJobCount() const { return m_job_count; }
    size_t GetMaxJobs() const { return m_max_jobs; }
    size_t GetScriptBlobUsed() const { return m_scriptpubkey_used + m_scriptsig_used; }
    size_t GetWitnessBlobUsed() const { return m_witness_used; }

    // GPU UTXO set reference (optional, for optimized lookups)
    void SetUTXOSet(GPUUTXOSet* utxo_set) { m_utxo_set = utxo_set; }

private:
    // Initialization state
    bool m_initialized;
    bool m_batch_active;

    // Job storage (host-side staging)
    std::vector<ScriptValidationJob> m_jobs;
    size_t m_job_count;
    size_t m_max_jobs;

    // Script data blobs (host-side staging)
    std::vector<uint8_t> m_scriptpubkey_blob;
    std::vector<uint8_t> m_scriptsig_blob;
    std::vector<uint8_t> m_witness_blob;
    size_t m_scriptpubkey_used;
    size_t m_scriptsig_used;
    size_t m_witness_used;
    size_t m_scriptpubkey_max;
    size_t m_scriptsig_max;
    size_t m_witness_max;

    // Device memory
    ScriptValidationJob* d_jobs;
    uint8_t* d_scriptpubkey_blob;
    uint8_t* d_scriptsig_blob;
    uint8_t* d_witness_blob;
    GPUScriptContext* d_contexts;  // Pre-allocated execution contexts (spilled to global memory)

    // Transaction data for sighash computation
    struct TxContext {
        uint256_gpu txid;
        uint256_gpu wtxid;
        int32_t version;
        uint32_t locktime;
        uint32_t num_inputs;
        uint32_t num_outputs;

        // Precomputed hashes (BIP143/341)
        uint256_gpu hashPrevouts;
        uint256_gpu hashSequence;
        uint256_gpu hashOutputs;
        uint256_gpu hashAmounts;
        uint256_gpu hashScriptPubKeys;
        bool hashes_computed;
    };
    std::vector<TxContext> m_tx_contexts;
    TxContext* d_tx_contexts;

    // Optional GPU UTXO set for optimized lookups
    GPUUTXOSet* m_utxo_set;

    // Internal helpers
    bool AllocateDeviceMemory();
    void FreeDeviceMemory();
    bool CopyToDevice();
    bool CopyFromDevice();

    // Determine sigversion from script type
    GPUSigVersion DetermineSigVersion(const uint8_t* scriptpubkey, uint32_t len,
                                      const uint8_t* witness, uint32_t witness_len) const;
};

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

// Main batch validation kernel - validates all queued script jobs
__global__ void BatchValidateScriptsKernel(
    ScriptValidationJob* jobs,
    const uint8_t* scriptpubkey_blob,
    const uint8_t* scriptsig_blob,
    const uint8_t* witness_blob,
    GPUScriptContext* contexts,
    uint32_t job_count
);

// Parallel sighash computation kernel
__global__ void BatchComputeSigHashKernel(
    ScriptValidationJob* jobs,
    const uint8_t* tx_data,
    uint32_t job_count
);

// Parallel signature verification kernel (ECDSA)
__global__ void BatchVerifyECDSAKernel(
    const uint8_t* sighashes,
    const uint8_t* signatures,
    const uint32_t* sig_sizes,
    const uint8_t* pubkeys,
    const uint32_t* pubkey_sizes,
    bool* results,
    uint32_t count
);

// Parallel signature verification kernel (Schnorr/Taproot)
__global__ void BatchVerifySchnorrKernel(
    const uint8_t* sighashes,
    const uint8_t* signatures,
    const uint8_t* pubkeys,
    bool* results,
    uint32_t count
);

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_BATCH_VALIDATOR_H
