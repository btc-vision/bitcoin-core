#ifndef BITCOIN_GPU_KERNEL_GPU_UTXO_H
#define BITCOIN_GPU_KERNEL_GPU_UTXO_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <cuda_runtime.h>
#include "gpu_types.h"

// Forward declarations to avoid circular dependencies
class uint256;
class CCoinsViewCache;

namespace gpu {

// Fixed-size UTXO header structure (32 bytes) for optimal GPU memory access
struct UTXOHeader {
    uint64_t amount;           // 8 bytes - satoshi amount
    uint64_t script_offset : 40;    // 5 bytes - offset into script blob (1TB addressable)
    uint64_t blockHeight : 24;      // 3 bytes - block height (16M blocks)
    uint32_t txid_index;       // 4 bytes - index into deduplicated txid table
    uint16_t script_size;      // 2 bytes - script length (up to 65KB)
    uint16_t vout;             // 2 bytes - output index in transaction
    uint8_t flags;             // 1 byte - coinbase, etc.
    uint8_t script_type;       // 1 byte - hint for fast-path validation
    uint8_t padding[6];        // 6 bytes - alignment to 32 bytes
};

static_assert(sizeof(UTXOHeader) == 32, "UTXOHeader must be exactly 32 bytes");

// Script types for fast validation paths
// Matches Bitcoin Core's TxoutType in script/solver.h
enum ScriptType : uint8_t {
    SCRIPT_TYPE_UNKNOWN = 0,
    SCRIPT_TYPE_P2PKH = 1,       // OP_DUP OP_HASH160 <20> OP_EQUALVERIFY OP_CHECKSIG
    SCRIPT_TYPE_P2WPKH = 2,      // OP_0 <20 bytes>
    SCRIPT_TYPE_P2SH = 3,        // OP_HASH160 <20> OP_EQUAL
    SCRIPT_TYPE_P2WSH = 4,       // OP_0 <32 bytes>
    SCRIPT_TYPE_P2TR = 5,        // OP_1 <32 bytes>
    SCRIPT_TYPE_P2PK = 6,        // <pubkey> OP_CHECKSIG (33 or 65 byte pubkey)
    SCRIPT_TYPE_MULTISIG = 7,    // OP_M <pubkey>... OP_N OP_CHECKMULTISIG
    SCRIPT_TYPE_NULL_DATA = 8,   // OP_RETURN <data> (provably unspendable)
    SCRIPT_TYPE_WITNESS_UNKNOWN = 9, // OP_N <2-40 bytes> where N >= 2
    SCRIPT_TYPE_NONSTANDARD = 255    // Any other valid script
};

// Flags for UTXO status
enum UTXOFlags : uint8_t {
    UTXO_FLAG_COINBASE = 0x01,
    UTXO_FLAG_SPENT = 0x02,
    UTXO_FLAG_DIRTY = 0x04
};

// Main GPU UTXO set class
class GPUUTXOSet {
public:
    // Constructor/Destructor
    GPUUTXOSet();
    ~GPUUTXOSet();

    // Initialize with specific memory limits
    bool Initialize(size_t maxVRAMUsage = 0);
    
    // Load UTXO set from CPU
    bool LoadFromCPU(const void* coinsCache);
    
    // Query operations
    bool HasUTXO(const uint256& txid, uint32_t vout) const;
    bool GetUTXO(const uint256& txid, uint32_t vout, UTXOHeader& header, uint8_t* scriptData = nullptr) const;
    
    // Accessor for hash tables (for kernels)
    uint32_t* GetHashTable(int index) { return d_hashTables[index]; }
    const uint32_t* GetHashTable(int index) const { return d_hashTables[index]; }
    
    // Update operations
    bool AddUTXO(const uint256& txid, uint32_t vout, const UTXOHeader& header, const uint8_t* scriptData);
    bool SpendUTXO(const uint256& txid, uint32_t vout);

    // =========================================================================
    // Batch operations for atomic reorg handling (Phase 8)
    // =========================================================================

    // Begin a batch update transaction - all subsequent operations are staged
    void BeginBatchUpdate();

    // Commit all staged changes atomically
    bool CommitBatchUpdate();

    // Abort batch update and discard all staged changes
    void AbortBatchUpdate();

    // Check if we're in a batch update
    bool IsInBatchUpdate() const { return m_batch_active; }

    // =========================================================================
    // Reorg-specific operations (Phase 8)
    // =========================================================================

    // Completely remove a UTXO (used when disconnecting blocks)
    // Different from SpendUTXO which just marks as spent
    bool RemoveUTXO(const uint256& txid, uint32_t vout);

    // Restore a spent UTXO from undo data (used when disconnecting blocks)
    // This unmarks the UTXO as spent and restores it to the hash table
    bool RestoreUTXO(const uint256& txid, uint32_t vout, const UTXOHeader& header, const uint8_t* scriptData);

    // Get UTXO even if spent (for undo operations)
    bool GetUTXOIncludingSpent(const uint256& txid, uint32_t vout, UTXOHeader& header, uint8_t* scriptData = nullptr) const;

    // Memory management
    size_t GetVRAMUsage() const { return totalVRAMUsed; }
    size_t GetFreeSpace() const { return totalFreeSpace; }
    bool NeedsCompaction() const { return totalFreeSpace > (totalVRAMUsed * 0.1); }
    bool Compact();

    // Statistics
    size_t GetNumUTXOs() const { return numUTXOs; }
    size_t GetMaxUTXOs() const { return maxUTXOs; }
    size_t GetScriptBlobUsed() const { return scriptBlobUsed; }
    size_t GetScriptBlobSize() const { return scriptBlobSize; }
    size_t GetTotalVRAMUsed() const { return totalVRAMUsed; }
    size_t GetTotalFreeSpace() const { return totalFreeSpace; }
    size_t GetTxidTableUsed() const { return txidTableUsed; }
    size_t GetTxidTableSize() const { return txidTableSize; }
    double GetLoadFactor() const;

    // =========================================================================
    // GPUDirect Storage Persistence (Direct GPU-to-Disk I/O)
    // =========================================================================

    // Save UTXO set directly from GPU memory to disk using GPUDirect Storage
    // This bypasses CPU entirely for maximum performance
    bool SaveToDisk(const std::string& datadir, uint64_t block_height, const uint8_t* block_hash);

    // Load UTXO set directly from disk to GPU memory using GPUDirect Storage
    bool LoadFromDisk(const std::string& datadir, uint64_t& block_height, uint8_t* block_hash);

    // Check if a valid UTXO snapshot exists on disk
    static bool HasDiskSnapshot(const std::string& datadir);

    // Get the block height of the on-disk snapshot (without loading)
    static bool GetDiskSnapshotHeight(const std::string& datadir, uint64_t& block_height);

    // =========================================================================
    // Incremental Persistence (for continuous disk updates)
    // =========================================================================

    // Flush pending changes to disk (call periodically or after batch commit)
    bool FlushToDisk();

    // Enable/disable automatic disk flushing after each block
    void SetAutoFlush(bool enabled) { m_auto_flush = enabled; }
    bool GetAutoFlush() const { return m_auto_flush; }

    // Get device pointers (for GPUDirect Storage direct access)
    UTXOHeader* GetHeadersPtr() { return d_headers; }
    const UTXOHeader* GetHeadersPtr() const { return d_headers; }
    uint8_t* GetScriptBlobPtr() { return d_scriptBlob; }
    const uint8_t* GetScriptBlobPtr() const { return d_scriptBlob; }
    uint256_gpu* GetTxidTablePtr() { return d_txidTable; }
    const uint256_gpu* GetTxidTablePtr() const { return d_txidTable; }

public:
    // Hash table operations (public for testing)
    uint32_t Hash1(const uint256& txid, uint32_t vout) const;
    uint32_t Hash2(const uint256& txid, uint32_t vout) const;
    uint32_t Hash3(const uint256& txid, uint32_t vout) const;
    uint32_t Hash4(const uint256& txid, uint32_t vout) const;
    
    // 4-way Cuckoo hash tables
    // Reduced to give priority to GPU script interpreter
    // 5M * 4 tables * 4 bytes = 80MB
    static constexpr size_t TABLE_SIZE = 5000000;  // 5M entries per table

private:
    // Device memory pointers
    UTXOHeader* d_headers;        // Fixed-size header array
    uint8_t* d_scriptBlob;        // Continuous script data blob
    uint256_gpu* d_txidTable;     // Deduplicated transaction IDs
    
    uint32_t* d_hashTables[4];   // 4 hash tables for cuckoo hashing
    
    // Host-side metadata
    size_t numUTXOs;
    size_t maxUTXOs;
    size_t scriptBlobSize;
    size_t scriptBlobUsed;
    size_t txidTableSize;
    size_t txidTableUsed;
    size_t totalVRAMUsed;
    size_t totalFreeSpace;
    size_t maxVRAMLimit;

    // =========================================================================
    // Batch update tracking (Phase 8 - Reorg handling)
    // =========================================================================
    bool m_batch_active{false};

    // Staged operations for batch commit using fixed-size arrays
    // Maximum capacity for batch operations (enough for largest blocks)
    static constexpr size_t MAX_STAGED_ADDS = 10000;
    static constexpr size_t MAX_STAGED_REMOVES = 10000;
    static constexpr size_t MAX_STAGED_RESTORES = 10000;
    static constexpr size_t MAX_SCRIPT_PER_STAGED = 520;  // Max script size

    struct StagedAdd {
        uint256_gpu txid;
        uint32_t vout;
        UTXOHeader header;
        uint8_t script_data[MAX_SCRIPT_PER_STAGED];
        uint16_t script_len;
    };

    struct StagedRemove {
        uint256_gpu txid;
        uint32_t vout;
        uint32_t utxo_index;  // Index in headers array
    };

    struct StagedRestore {
        uint256_gpu txid;
        uint32_t vout;
        uint32_t utxo_index;  // Index of previously spent UTXO
    };

    // Host-side staging arrays (allocated lazily)
    StagedAdd* m_staged_adds{nullptr};
    StagedRemove* m_staged_removes{nullptr};
    StagedRestore* m_staged_restores{nullptr};
    size_t m_staged_adds_count{0};
    size_t m_staged_removes_count{0};
    size_t m_staged_restores_count{0};

    // Snapshot of state before batch for rollback
    size_t m_snapshot_numUTXOs{0};
    size_t m_snapshot_scriptBlobUsed{0};
    size_t m_snapshot_txidTableUsed{0};

    // GPUDirect Storage persistence
    bool m_auto_flush{false};
    std::string m_datadir;
    uint64_t m_last_flushed_height{0};

    // =========================================================================
    // Incremental flushing tracking
    // =========================================================================

    // Track the state at last flush for incremental updates
    size_t m_flush_numUTXOs{0};           // numUTXOs at last flush
    size_t m_flush_scriptBlobUsed{0};     // scriptBlobUsed at last flush
    size_t m_flush_txidTableUsed{0};      // txidTableUsed at last flush

    // Track dirty UTXO indices (UTXOs modified since last flush)
    // These are indices of existing UTXOs that were modified (e.g., spent)
    uint32_t* m_dirty_indices{nullptr};
    size_t m_dirty_count{0};
    size_t m_dirty_capacity{0};
    static constexpr size_t DIRTY_TRACK_INCREMENT = 10000;

    // Helper to track a modified UTXO index
    void MarkDirty(uint32_t index);
    void ClearDirtyTracking();

    // Memory allocation helpers
    bool AllocateDeviceMemory();
    void FreeDeviceMemory();
    
    // Txid table management
    uint32_t GetOrAddTxid(const uint256& txid);
    
    // Script blob management
    uint64_t AllocateScriptSpace(size_t size);
    void FreeScriptSpace(uint64_t offset, size_t size);
};

// CUDA kernel declarations
__global__ void FindUTXOKernel(
    const UTXOHeader* headers,
    const uint256_gpu* txidTable,
    const uint32_t* hashTables,
    const uint256_gpu* searchTxid,
    uint32_t searchVout,
    uint32_t* resultIndex
);

__global__ void ValidateP2PKHKernel(
    const UTXOHeader* headers,
    const uint8_t* scriptBlob,
    const uint32_t* utxoIndices,
    const uint8_t* signatures,
    const uint8_t* pubkeys,
    bool* results,
    uint32_t count
);

__global__ void CompactScriptBlobKernel(
    uint8_t* scriptBlob,
    const uint64_t* oldOffsets,
    const uint64_t* newOffsets,
    const uint16_t* sizes,
    uint32_t count
);

// Helper functions
size_t GetAvailableVRAM();
ScriptType IdentifyScriptType(const uint8_t* script, size_t size);

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_UTXO_H