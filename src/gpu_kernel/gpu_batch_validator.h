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

// Ensure proper alignment for GPU memory access
struct alignas(8) ScriptValidationJob {
    // 8-byte aligned fields first
    int64_t amount;                 // Amount being spent (satoshis) - 8 bytes

    // 32-byte aligned sighash
    uint256_gpu sighash;            // Precomputed sighash (when available) - 32 bytes

    // 4-byte aligned fields
    uint32_t tx_index;              // Index in block
    uint32_t input_index;           // Input index within transaction
    uint32_t scriptpubkey_offset;   // Offset in scriptPubKey blob
    uint32_t scriptsig_offset;      // Offset in scriptSig blob
    uint32_t witness_offset;        // Offset in witness blob
    uint32_t sequence;              // Input sequence number
    uint32_t verify_flags;          // SCRIPT_VERIFY_* flags

    // 2-byte aligned fields (grouped together)
    uint16_t scriptpubkey_size;     // Size of scriptPubKey
    uint16_t scriptsig_size;        // Size of scriptSig
    uint16_t witness_count;         // Number of witness stack items
    uint32_t witness_total_size;    // Total witness data size (needs 32 bits for large tapscripts)

    // Enums (typically 4 bytes each)
    GPUSigVersion sigversion;       // Signature version
    GPUScriptError error;           // Error code if validation failed

    // 1-byte fields (grouped at end)
    bool sighash_valid;             // Whether sighash is precomputed
    bool validated;                 // Whether validation was attempted
    bool valid;                     // Whether validation passed
    uint8_t padding[1];             // Explicit padding for alignment
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
    // NOTE: GPUScriptContext is ~1.05MB each (2x stack of 1000 * 524 bytes)
    // With typical 2-8GB available for batch validator after UTXO set allocation,
    // we dynamically calculate max_jobs based on available memory.
    // 2000 jobs = ~2.1GB which is the default target.
    // Blocks have ~2000-4000 inputs, so larger batches reduce kernel launches.
    static constexpr size_t DEFAULT_MAX_JOBS = 2000;   // ~2.1GB VRAM for contexts
    static constexpr size_t MIN_MAX_JOBS = 100;        // Minimum useful batch size
    static constexpr size_t CONTEXT_SIZE_BYTES = 1100000;  // ~1.05MB per GPUScriptContext
    static constexpr size_t DEFAULT_SCRIPT_BLOB_SIZE = 16 * 1024 * 1024;  // 16MB
    static constexpr size_t DEFAULT_WITNESS_BLOB_SIZE = 32 * 1024 * 1024; // 32MB

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
    // sighash: precomputed signature hash (32 bytes), or nullptr if not available
    int QueueJob(
        uint32_t tx_index,
        uint32_t input_index,
        const uint8_t* scriptpubkey, uint32_t scriptpubkey_len,
        const uint8_t* scriptsig, uint32_t scriptsig_len,
        const uint8_t* witness, uint32_t witness_len, uint32_t witness_count,
        int64_t amount,
        uint32_t sequence,
        uint32_t verify_flags,
        GPUSigVersion sigversion,
        const uint8_t* sighash = nullptr  // 32-byte precomputed sighash, or nullptr
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

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_BATCH_VALIDATOR_H
