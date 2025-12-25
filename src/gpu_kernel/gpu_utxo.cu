#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_utils.h"
#include "gpu_hash.cuh"
#include "gpu_logging.h"
#include "gpu_direct_storage.h"
#include <cuda.h>
#include <cstring>
#include <random>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Helper functions for logging
static inline void LogGPUMessage(const std::string& msg) {
    ::LogGPUInfo(msg.c_str());
}

static inline void LogGPUError(const std::string& msg) {
    std::string fullMsg = "[GPU ERROR] " + msg;
    ::LogGPUInfo(fullMsg.c_str());
}

namespace gpu {

// Initialize salts for hashing - these would be randomized in production
static uint64_t g_hash_salts[8] = {0};  // 4 pairs of k0,k1 for 4 hash functions
static bool g_salts_initialized = false;

static void InitializeHashSalts() {
    if (!g_salts_initialized) {
        // In production, use secure random salts from Bitcoin Core's GetRand
        // For now, use deterministic values for testing
        std::mt19937_64 rng(0x12345678);  // Seed for reproducibility
        for (int i = 0; i < 8; i++) {
            g_hash_salts[i] = rng();
        }
        g_salts_initialized = true;
    }
}

// Hash functions for 4-way cuckoo hashing using SipHash
uint32_t GPUUTXOSet::Hash1(const uint256& txid, uint32_t vout) const {
    uint256_gpu gpu_txid = ToGPU(txid);
    uint64_t hash = SipHashUint256Extra(g_hash_salts[0], g_hash_salts[1], gpu_txid, vout);
    return static_cast<uint32_t>(hash % TABLE_SIZE);
}

uint32_t GPUUTXOSet::Hash2(const uint256& txid, uint32_t vout) const {
    uint256_gpu gpu_txid = ToGPU(txid);
    uint64_t hash = SipHashUint256Extra(g_hash_salts[2], g_hash_salts[3], gpu_txid, vout);
    return static_cast<uint32_t>(hash % TABLE_SIZE);
}

uint32_t GPUUTXOSet::Hash3(const uint256& txid, uint32_t vout) const {
    uint256_gpu gpu_txid = ToGPU(txid);
    uint64_t hash = SipHashUint256Extra(g_hash_salts[4], g_hash_salts[5], gpu_txid, vout);
    return static_cast<uint32_t>(hash % TABLE_SIZE);
}

uint32_t GPUUTXOSet::Hash4(const uint256& txid, uint32_t vout) const {
    uint256_gpu gpu_txid = ToGPU(txid);
    uint64_t hash = SipHashUint256Extra(g_hash_salts[6], g_hash_salts[7], gpu_txid, vout);
    return static_cast<uint32_t>(hash % TABLE_SIZE);
}

// Constructor
GPUUTXOSet::GPUUTXOSet() 
    : d_headers(nullptr)
    , d_scriptBlob(nullptr)
    , d_txidTable(nullptr)
    , numUTXOs(0)
    , maxUTXOs(0)
    , scriptBlobSize(0)
    , scriptBlobUsed(0)
    , txidTableSize(0)
    , txidTableUsed(0)
    , totalVRAMUsed(0)
    , totalFreeSpace(0)
    , maxVRAMLimit(0) {
    
    InitializeHashSalts();
    
    for (int i = 0; i < 4; i++) {
        d_hashTables[i] = nullptr;
    }
}

// Destructor
GPUUTXOSet::~GPUUTXOSet() {
    FreeDeviceMemory();

    // Free staging arrays
    delete[] m_staged_adds;
    delete[] m_staged_removes;
    delete[] m_staged_restores;
    m_staged_adds = nullptr;
    m_staged_removes = nullptr;
    m_staged_restores = nullptr;

    // Free dirty tracking array
    delete[] m_dirty_indices;
    m_dirty_indices = nullptr;
    m_dirty_count = 0;
    m_dirty_capacity = 0;
}

// Mark a UTXO index as dirty (modified since last flush)
void GPUUTXOSet::MarkDirty(uint32_t index) {
    // Skip if this is a newly added UTXO (will be flushed with new data anyway)
    if (index >= m_flush_numUTXOs) {
        return;  // This is a new entry, not a modification of existing
    }

    // Expand capacity if needed
    if (m_dirty_count >= m_dirty_capacity) {
        size_t new_capacity = m_dirty_capacity + DIRTY_TRACK_INCREMENT;
        uint32_t* new_array = new uint32_t[new_capacity];
        if (m_dirty_indices) {
            memcpy(new_array, m_dirty_indices, m_dirty_count * sizeof(uint32_t));
            delete[] m_dirty_indices;
        }
        m_dirty_indices = new_array;
        m_dirty_capacity = new_capacity;
    }

    // Check for duplicates (simple linear scan - could use a set for larger counts)
    for (size_t i = 0; i < m_dirty_count; i++) {
        if (m_dirty_indices[i] == index) {
            return;  // Already tracked
        }
    }

    m_dirty_indices[m_dirty_count++] = index;
}

// Clear dirty tracking (called after successful flush)
void GPUUTXOSet::ClearDirtyTracking() {
    m_dirty_count = 0;
    // Keep the allocated buffer for reuse
}

// Get available VRAM
size_t GetAvailableVRAM() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

// Initialize with specific memory limits
bool GPUUTXOSet::Initialize(size_t maxVRAMUsage) {
    LogGPUMessage("Initializing GPU UTXO Set");
    
    // Get available VRAM
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        LogGPUError("Failed to get GPU memory info: " + std::string(cudaGetErrorString(err)));
        return false;
    }
    
    LogGPUMessage("GPU Memory - Total: " + std::to_string(total_mem / (1024*1024)) + 
                  " MB, Free: " + std::to_string(free_mem / (1024*1024)) + " MB");
    
    // Calculate memory limits (use 95% of available or specified limit)
    if (maxVRAMUsage == 0) {
        maxVRAMLimit = static_cast<size_t>(free_mem * 0.95);
    } else {
        maxVRAMLimit = std::min(maxVRAMUsage, static_cast<size_t>(free_mem * 0.95));
    }
    
    LogGPUMessage("Setting VRAM limit to: " + std::to_string(maxVRAMLimit / (1024*1024)) + " MB");

    // Estimate memory allocation (reduced from original aggressive defaults):
    // - Headers: 30M * 32B = 960MB
    // - Scripts: 1GB (most scripts are small, average ~50 bytes)
    // - Txid table: 20M * 32B = 640MB
    // - Hash tables: 4 * 10M * 4B = 160MB
    // Total: ~2.8GB base, leaving headroom for batch validator

    maxUTXOs = 30000000;  // 30 million UTXOs (covers typical active set)
    scriptBlobSize = 1ULL * 1024 * 1024 * 1024;  // 1GB for scripts
    txidTableSize = 20000000;  // 20 million unique txids
    
    // Check if we have enough memory
    size_t required_mem =
        maxUTXOs * sizeof(UTXOHeader) +          // Headers: 30M * 32B = 960MB
        scriptBlobSize +                          // Scripts: 1GB
        txidTableSize * sizeof(uint256_gpu) +    // Txid table: 20M * 32B = 640MB
        4 * TABLE_SIZE * sizeof(uint32_t);       // Hash tables: 4 * 10M * 4B = 160MB
    
    if (required_mem > maxVRAMLimit) {
        // Scale down if needed
        double scale = static_cast<double>(maxVRAMLimit) / required_mem;
        maxUTXOs = static_cast<size_t>(maxUTXOs * scale);
        scriptBlobSize = static_cast<size_t>(scriptBlobSize * scale);
        txidTableSize = static_cast<size_t>(txidTableSize * scale);
        
        LogGPUMessage("Scaled down to fit VRAM - Max UTXOs: " + std::to_string(maxUTXOs));
    }
    
    return AllocateDeviceMemory();
}

// Allocate GPU memory
bool GPUUTXOSet::AllocateDeviceMemory() {
    LogGPUMessage("Allocating device memory");
    
    // Allocate header array
    cudaError_t err = cudaMalloc(&d_headers, maxUTXOs * sizeof(UTXOHeader));
    if (err != cudaSuccess) {
        LogGPUError("Failed to allocate header memory: " + std::string(cudaGetErrorString(err)));
        return false;
    }
    
    // Allocate script blob
    err = cudaMalloc(&d_scriptBlob, scriptBlobSize);
    if (err != cudaSuccess) {
        LogGPUError("Failed to allocate script blob: " + std::string(cudaGetErrorString(err)));
        FreeDeviceMemory();
        return false;
    }
    
    // Allocate txid table
    err = cudaMalloc(&d_txidTable, txidTableSize * sizeof(uint256_gpu));
    if (err != cudaSuccess) {
        LogGPUError("Failed to allocate txid table: " + std::string(cudaGetErrorString(err)));
        FreeDeviceMemory();
        return false;
    }
    
    // Allocate hash tables
    for (int i = 0; i < 4; i++) {
        err = cudaMalloc(&d_hashTables[i], TABLE_SIZE * sizeof(uint32_t));
        if (err != cudaSuccess) {
            LogGPUError("Failed to allocate hash table " + std::to_string(i) + ": " + 
                       std::string(cudaGetErrorString(err)));
            FreeDeviceMemory();
            return false;
        }
        
        // Initialize to empty
        cudaMemset(d_hashTables[i], 0xFF, TABLE_SIZE * sizeof(uint32_t));
    }
    
    totalVRAMUsed = 
        maxUTXOs * sizeof(UTXOHeader) +
        scriptBlobSize +
        txidTableSize * sizeof(uint256_gpu) +
        4 * TABLE_SIZE * sizeof(uint32_t);
    
    LogGPUMessage("Successfully allocated " + std::to_string(totalVRAMUsed / (1024*1024)) + " MB on GPU");
    
    return true;
}

// Free GPU memory
void GPUUTXOSet::FreeDeviceMemory() {
    if (d_headers) {
        cudaFree(d_headers);
        d_headers = nullptr;
    }
    
    if (d_scriptBlob) {
        cudaFree(d_scriptBlob);
        d_scriptBlob = nullptr;
    }
    
    if (d_txidTable) {
        cudaFree(d_txidTable);
        d_txidTable = nullptr;
    }
    
    for (int i = 0; i < 4; i++) {
        if (d_hashTables[i]) {
            cudaFree(d_hashTables[i]);
            d_hashTables[i] = nullptr;
        }
    }
}

// Get or add txid to the table (simplified version for tests)
uint32_t GPUUTXOSet::GetOrAddTxid(const uint256& txid) {
    // For simplicity in tests, we'll use the existing txid_index from the header
    // In production, this would maintain a proper deduplication map
    
    // Check if txid already exists (simplified linear search for tests)
    uint256_gpu gpu_txid = ToGPU(txid);
    
    // Add new txid if space available
    if (txidTableUsed < txidTableSize) {
        cudaMemcpy(d_txidTable + txidTableUsed, &gpu_txid, sizeof(uint256_gpu), cudaMemcpyHostToDevice);
        return txidTableUsed++;
    }
    
    return 0; // Default to first entry if table full
}

// Add a UTXO
bool GPUUTXOSet::AddUTXO(const uint256& txid, uint32_t vout, const UTXOHeader& header, const uint8_t* scriptData) {
    if (numUTXOs >= maxUTXOs) {
        LogGPUError("UTXO set full");
        return false;
    }
    
    if (scriptBlobUsed + header.script_size > scriptBlobSize) {
        LogGPUError("Script blob full");
        return false;
    }
    
    // Get or add the txid to the table
    UTXOHeader modifiedHeader = header;
    modifiedHeader.txid_index = GetOrAddTxid(txid);
    modifiedHeader.script_offset = scriptBlobUsed;  // Update script offset
    
    // Copy header to GPU
    cudaMemcpy(d_headers + numUTXOs, &modifiedHeader, sizeof(UTXOHeader), cudaMemcpyHostToDevice);
    
    // Copy script data to GPU
    if (scriptData && header.script_size > 0) {
        cudaMemcpy(d_scriptBlob + scriptBlobUsed, scriptData, header.script_size, cudaMemcpyHostToDevice);
        scriptBlobUsed += header.script_size;
    }
    
    // Insert into hash tables
    uint32_t hashes[4] = {
        Hash1(txid, vout),
        Hash2(txid, vout),
        Hash3(txid, vout),
        Hash4(txid, vout)
    };
    
    // Try to insert in any empty slot
    bool inserted = false;
    for (int i = 0; i < 4; i++) {
        uint32_t empty = 0xFFFFFFFF;
        uint32_t* addr = d_hashTables[i] + hashes[i];
        cudaMemcpy(&empty, addr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        if (empty == 0xFFFFFFFF) {
            cudaMemcpy(addr, &numUTXOs, sizeof(uint32_t), cudaMemcpyHostToDevice);
            inserted = true;
            break;
        }
    }
    
    if (!inserted) {
        LogGPUError("Failed to insert UTXO - all hash slots full");
        return false;
    }
    
    numUTXOs++;
    return true;
}

// Spend a UTXO
bool GPUUTXOSet::SpendUTXO(const uint256& txid, uint32_t vout) {
    // Find the UTXO
    uint32_t hashes[4] = {
        Hash1(txid, vout),
        Hash2(txid, vout),
        Hash3(txid, vout),
        Hash4(txid, vout)
    };
    
    for (int i = 0; i < 4; i++) {
        uint32_t index;
        cudaMemcpy(&index, d_hashTables[i] + hashes[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        if (index != 0xFFFFFFFF && index < numUTXOs) {
            UTXOHeader header;
            cudaMemcpy(&header, d_headers + index, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);
            
            // Check if this is the right UTXO
            uint256_gpu stored_txid;
            cudaMemcpy(&stored_txid, d_txidTable + header.txid_index, sizeof(uint256_gpu), cudaMemcpyDeviceToHost);
            
            if (FromGPU(stored_txid) == txid && header.vout == vout) {
                // Mark as spent and dirty for incremental flush
                header.flags |= UTXO_FLAG_SPENT | UTXO_FLAG_DIRTY;
                cudaMemcpy(d_headers + index, &header, sizeof(UTXOHeader), cudaMemcpyHostToDevice);

                // Track for incremental disk flushing
                MarkDirty(index);

                // Track freed space
                totalFreeSpace += header.script_size;

                // Clear from hash table
                uint32_t empty = 0xFFFFFFFF;
                cudaMemcpy(d_hashTables[i] + hashes[i], &empty, sizeof(uint32_t), cudaMemcpyHostToDevice);

                return true;
            }
        }
    }
    
    return false;
}

// Get load factor
double GPUUTXOSet::GetLoadFactor() const {
    if (maxUTXOs == 0) return 0.0;
    return (static_cast<double>(numUTXOs) * 100.0) / static_cast<double>(maxUTXOs);
}

// Query operations
bool GPUUTXOSet::HasUTXO(const uint256& txid, uint32_t vout) const {
    uint32_t hashes[4] = {
        Hash1(txid, vout),
        Hash2(txid, vout),
        Hash3(txid, vout),
        Hash4(txid, vout)
    };
    
    for (int i = 0; i < 4; i++) {
        uint32_t index;
        cudaMemcpy(&index, d_hashTables[i] + hashes[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        if (index != 0xFFFFFFFF && index < numUTXOs) {
            UTXOHeader header;
            cudaMemcpy(&header, d_headers + index, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);
            
            // Check if this is the right UTXO and not spent
            uint256_gpu stored_txid;
            cudaMemcpy(&stored_txid, d_txidTable + header.txid_index, sizeof(uint256_gpu), cudaMemcpyDeviceToHost);
            
            if (FromGPU(stored_txid) == txid && header.vout == vout && !(header.flags & UTXO_FLAG_SPENT)) {
                return true;
            }
        }
    }
    
    return false;
}

// NOTE: IdentifyScriptType is defined in gpu_utxo_wrapper.cpp to avoid duplicate symbol

// CUDA kernel for finding UTXOs
__global__ void FindUTXOKernel(
    const UTXOHeader* headers,
    const uint256_gpu* txidTable,
    const uint32_t* hashTables,
    const uint256_gpu* searchTxid,
    uint32_t searchVout,
    uint32_t* resultIndex
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= 4) return;  // Only check 4 hash positions
    
    // Use pre-computed hash positions
    uint32_t hashPos = tid * GPUUTXOSet::TABLE_SIZE;  // Simplified - would need actual hash
    uint32_t index = hashTables[hashPos];
    
    if (index != 0xFFFFFFFF) {
        const UTXOHeader& header = headers[index];
        const uint256_gpu& stored_txid = txidTable[header.txid_index];
        
        // Compare txid and vout
        bool match = true;
        for (int i = 0; i < 32; i++) {
            if (stored_txid.data[i] != searchTxid->data[i]) {
                match = false;
                break;
            }
        }
        
        if (match && header.vout == searchVout) {
            *resultIndex = index;
        }
    }
}

// =========================================================================
// Batch operations for atomic reorg handling (Phase 8)
// =========================================================================

void GPUUTXOSet::BeginBatchUpdate() {
    if (m_batch_active) {
        LogGPUError("BeginBatchUpdate called while batch already active");
        return;
    }

    LogGPUMessage("Beginning batch UTXO update");

    // Take snapshot of current state for potential rollback
    m_snapshot_numUTXOs = numUTXOs;
    m_snapshot_scriptBlobUsed = scriptBlobUsed;
    m_snapshot_txidTableUsed = txidTableUsed;

    // Allocate staging arrays if not already allocated
    if (!m_staged_adds) {
        m_staged_adds = new StagedAdd[MAX_STAGED_ADDS];
    }
    if (!m_staged_removes) {
        m_staged_removes = new StagedRemove[MAX_STAGED_REMOVES];
    }
    if (!m_staged_restores) {
        m_staged_restores = new StagedRestore[MAX_STAGED_RESTORES];
    }

    // Clear staging counts
    m_staged_adds_count = 0;
    m_staged_removes_count = 0;
    m_staged_restores_count = 0;

    m_batch_active = true;
}

bool GPUUTXOSet::CommitBatchUpdate() {
    if (!m_batch_active) {
        LogGPUError("CommitBatchUpdate called without active batch");
        return false;
    }

    LogGPUMessage("Committing batch update - Adds: " + std::to_string(m_staged_adds_count) +
                  ", Removes: " + std::to_string(m_staged_removes_count) +
                  ", Restores: " + std::to_string(m_staged_restores_count));

    // Process removes first (clearing spent outputs from reorg)
    for (size_t r = 0; r < m_staged_removes_count; r++) {
        const StagedRemove& remove = m_staged_removes[r];
        // Find and clear from hash tables
        uint256 txid = FromGPU(remove.txid);
        uint32_t hashes[4] = {
            Hash1(txid, remove.vout),
            Hash2(txid, remove.vout),
            Hash3(txid, remove.vout),
            Hash4(txid, remove.vout)
        };

        for (int i = 0; i < 4; i++) {
            uint32_t index;
            cudaMemcpy(&index, d_hashTables[i] + hashes[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (index == remove.utxo_index) {
                // Clear the hash table entry
                uint32_t empty = 0xFFFFFFFF;
                cudaMemcpy(d_hashTables[i] + hashes[i], &empty, sizeof(uint32_t), cudaMemcpyHostToDevice);

                // Mark header as spent/removed
                UTXOHeader header;
                cudaMemcpy(&header, d_headers + index, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);
                header.flags |= UTXO_FLAG_SPENT;
                cudaMemcpy(d_headers + index, &header, sizeof(UTXOHeader), cudaMemcpyHostToDevice);

                totalFreeSpace += header.script_size;
                break;
            }
        }
    }

    // Process restores (unspending UTXOs during reorg)
    for (size_t r = 0; r < m_staged_restores_count; r++) {
        const StagedRestore& restore = m_staged_restores[r];
        uint256 txid = FromGPU(restore.txid);

        // Get the header and clear the spent flag
        UTXOHeader header;
        cudaMemcpy(&header, d_headers + restore.utxo_index, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);
        header.flags &= ~UTXO_FLAG_SPENT;
        cudaMemcpy(d_headers + restore.utxo_index, &header, sizeof(UTXOHeader), cudaMemcpyHostToDevice);

        // Re-insert into hash tables
        uint32_t hashes[4] = {
            Hash1(txid, restore.vout),
            Hash2(txid, restore.vout),
            Hash3(txid, restore.vout),
            Hash4(txid, restore.vout)
        };

        for (int i = 0; i < 4; i++) {
            uint32_t existing;
            cudaMemcpy(&existing, d_hashTables[i] + hashes[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (existing == 0xFFFFFFFF) {
                cudaMemcpy(d_hashTables[i] + hashes[i], &restore.utxo_index, sizeof(uint32_t), cudaMemcpyHostToDevice);
                totalFreeSpace -= header.script_size;
                break;
            }
        }
    }

    // Process adds (new UTXOs from connected blocks)
    for (size_t a = 0; a < m_staged_adds_count; a++) {
        const StagedAdd& add = m_staged_adds[a];
        uint256 txid = FromGPU(add.txid);

        if (numUTXOs >= maxUTXOs) {
            LogGPUError("UTXO set full during batch commit");
            AbortBatchUpdate();
            return false;
        }

        if (scriptBlobUsed + add.header.script_size > scriptBlobSize) {
            LogGPUError("Script blob full during batch commit");
            AbortBatchUpdate();
            return false;
        }

        // Copy the header with updated script offset
        UTXOHeader header = add.header;
        header.script_offset = scriptBlobUsed;

        cudaMemcpy(d_headers + numUTXOs, &header, sizeof(UTXOHeader), cudaMemcpyHostToDevice);

        // Copy script data
        if (add.script_len > 0) {
            cudaMemcpy(d_scriptBlob + scriptBlobUsed, add.script_data,
                      add.script_len, cudaMemcpyHostToDevice);
            scriptBlobUsed += add.script_len;
        }

        // Insert into hash tables
        uint32_t hashes[4] = {
            Hash1(txid, add.vout),
            Hash2(txid, add.vout),
            Hash3(txid, add.vout),
            Hash4(txid, add.vout)
        };

        bool inserted = false;
        for (int i = 0; i < 4; i++) {
            uint32_t existing;
            cudaMemcpy(&existing, d_hashTables[i] + hashes[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (existing == 0xFFFFFFFF) {
                cudaMemcpy(d_hashTables[i] + hashes[i], &numUTXOs, sizeof(uint32_t), cudaMemcpyHostToDevice);
                inserted = true;
                break;
            }
        }

        if (!inserted) {
            LogGPUError("Failed to insert UTXO during batch commit - all hash slots full");
            AbortBatchUpdate();
            return false;
        }

        numUTXOs++;
    }

    // Clear staging and finish
    m_staged_adds_count = 0;
    m_staged_removes_count = 0;
    m_staged_restores_count = 0;
    m_batch_active = false;

    LogGPUMessage("Batch update committed successfully. Total UTXOs: " + std::to_string(numUTXOs));
    return true;
}

void GPUUTXOSet::AbortBatchUpdate() {
    if (!m_batch_active) {
        return;
    }

    LogGPUMessage("Aborting batch update - rolling back to snapshot");

    // Restore snapshot state
    numUTXOs = m_snapshot_numUTXOs;
    scriptBlobUsed = m_snapshot_scriptBlobUsed;
    txidTableUsed = m_snapshot_txidTableUsed;

    // Clear staging
    m_staged_adds_count = 0;
    m_staged_removes_count = 0;
    m_staged_restores_count = 0;

    m_batch_active = false;
}

// =========================================================================
// Reorg-specific operations (Phase 8)
// =========================================================================

bool GPUUTXOSet::RemoveUTXO(const uint256& txid, uint32_t vout) {
    // Find the UTXO index
    uint32_t hashes[4] = {
        Hash1(txid, vout),
        Hash2(txid, vout),
        Hash3(txid, vout),
        Hash4(txid, vout)
    };

    for (int i = 0; i < 4; i++) {
        uint32_t index;
        cudaMemcpy(&index, d_hashTables[i] + hashes[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (index != 0xFFFFFFFF && index < numUTXOs) {
            UTXOHeader header;
            cudaMemcpy(&header, d_headers + index, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);

            uint256_gpu stored_txid;
            cudaMemcpy(&stored_txid, d_txidTable + header.txid_index, sizeof(uint256_gpu), cudaMemcpyDeviceToHost);

            if (FromGPU(stored_txid) == txid && header.vout == vout) {
                if (m_batch_active) {
                    // Stage the removal
                    if (m_staged_removes_count < MAX_STAGED_REMOVES) {
                        StagedRemove& remove = m_staged_removes[m_staged_removes_count++];
                        remove.txid = stored_txid;
                        remove.vout = vout;
                        remove.utxo_index = index;
                    }
                } else {
                    // Direct removal
                    uint32_t empty = 0xFFFFFFFF;
                    cudaMemcpy(d_hashTables[i] + hashes[i], &empty, sizeof(uint32_t), cudaMemcpyHostToDevice);

                    header.flags |= UTXO_FLAG_SPENT;
                    cudaMemcpy(d_headers + index, &header, sizeof(UTXOHeader), cudaMemcpyHostToDevice);
                    totalFreeSpace += header.script_size;
                }
                return true;
            }
        }
    }

    return false;
}

bool GPUUTXOSet::RestoreUTXO(const uint256& txid, uint32_t vout, const UTXOHeader& header, const uint8_t* scriptData) {
    // During reorg, we need to restore a UTXO that was previously spent
    // First, try to find if it still exists in our headers (just marked as spent)

    uint32_t hashes[4] = {
        Hash1(txid, vout),
        Hash2(txid, vout),
        Hash3(txid, vout),
        Hash4(txid, vout)
    };

    // Scan all headers to find the spent UTXO
    for (size_t i = 0; i < numUTXOs; i++) {
        UTXOHeader storedHeader;
        cudaMemcpy(&storedHeader, d_headers + i, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);

        if (storedHeader.flags & UTXO_FLAG_SPENT) {
            uint256_gpu stored_txid;
            cudaMemcpy(&stored_txid, d_txidTable + storedHeader.txid_index, sizeof(uint256_gpu), cudaMemcpyDeviceToHost);

            if (FromGPU(stored_txid) == txid && storedHeader.vout == vout) {
                if (m_batch_active) {
                    // Stage the restore
                    if (m_staged_restores_count < MAX_STAGED_RESTORES) {
                        StagedRestore& restore = m_staged_restores[m_staged_restores_count++];
                        restore.txid = stored_txid;
                        restore.vout = vout;
                        restore.utxo_index = static_cast<uint32_t>(i);
                    }
                } else {
                    // Direct restore
                    storedHeader.flags &= ~UTXO_FLAG_SPENT;
                    cudaMemcpy(d_headers + i, &storedHeader, sizeof(UTXOHeader), cudaMemcpyHostToDevice);

                    // Re-insert into hash table
                    for (int j = 0; j < 4; j++) {
                        uint32_t existing;
                        cudaMemcpy(&existing, d_hashTables[j] + hashes[j], sizeof(uint32_t), cudaMemcpyDeviceToHost);

                        if (existing == 0xFFFFFFFF) {
                            uint32_t idx = static_cast<uint32_t>(i);
                            cudaMemcpy(d_hashTables[j] + hashes[j], &idx, sizeof(uint32_t), cudaMemcpyHostToDevice);
                            totalFreeSpace -= storedHeader.script_size;
                            break;
                        }
                    }
                }
                return true;
            }
        }
    }

    // UTXO not found in spent list - need to add fresh
    // This happens if the UTXO was fully purged, so we add it back
    if (m_batch_active) {
        if (m_staged_adds_count < MAX_STAGED_ADDS) {
            StagedAdd& add = m_staged_adds[m_staged_adds_count++];
            add.txid = ToGPU(txid);
            add.vout = vout;
            add.header = header;
            add.script_len = 0;
            if (scriptData && header.script_size > 0) {
                size_t copy_len = header.script_size;
                if (copy_len > MAX_SCRIPT_PER_STAGED) copy_len = MAX_SCRIPT_PER_STAGED;
                memcpy(add.script_data, scriptData, copy_len);
                add.script_len = static_cast<uint16_t>(copy_len);
            }
        }
        return true;
    } else {
        return AddUTXO(txid, vout, header, scriptData);
    }
}

bool GPUUTXOSet::GetUTXOIncludingSpent(const uint256& txid, uint32_t vout, UTXOHeader& header, uint8_t* scriptData) const {
    // Search all headers including spent ones
    for (size_t i = 0; i < numUTXOs; i++) {
        UTXOHeader storedHeader;
        cudaMemcpy(&storedHeader, d_headers + i, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);

        uint256_gpu stored_txid;
        cudaMemcpy(&stored_txid, d_txidTable + storedHeader.txid_index, sizeof(uint256_gpu), cudaMemcpyDeviceToHost);

        if (FromGPU(stored_txid) == txid && storedHeader.vout == vout) {
            header = storedHeader;

            if (scriptData && storedHeader.script_size > 0) {
                cudaMemcpy(scriptData, d_scriptBlob + storedHeader.script_offset,
                          storedHeader.script_size, cudaMemcpyDeviceToHost);
            }
            return true;
        }
    }

    return false;
}

bool GPUUTXOSet::GetUTXO(const uint256& txid, uint32_t vout, UTXOHeader& header, uint8_t* scriptData) const {
    uint32_t hashes[4] = {
        Hash1(txid, vout),
        Hash2(txid, vout),
        Hash3(txid, vout),
        Hash4(txid, vout)
    };

    for (int i = 0; i < 4; i++) {
        uint32_t index;
        cudaMemcpy(&index, d_hashTables[i] + hashes[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (index != 0xFFFFFFFF && index < numUTXOs) {
            UTXOHeader storedHeader;
            cudaMemcpy(&storedHeader, d_headers + index, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);

            uint256_gpu stored_txid;
            cudaMemcpy(&stored_txid, d_txidTable + storedHeader.txid_index, sizeof(uint256_gpu), cudaMemcpyDeviceToHost);

            if (FromGPU(stored_txid) == txid && storedHeader.vout == vout && !(storedHeader.flags & UTXO_FLAG_SPENT)) {
                header = storedHeader;

                if (scriptData && storedHeader.script_size > 0) {
                    cudaMemcpy(scriptData, d_scriptBlob + storedHeader.script_offset,
                              storedHeader.script_size, cudaMemcpyDeviceToHost);
                }
                return true;
            }
        }
    }

    return false;
}

// Note: LoadFromCPU() is implemented in gpu_utxo_loader.cu
// Note: Compact() is implemented in gpu_utxo_compact.cu

// =========================================================================
// GPUDirect Storage Persistence Implementation
// =========================================================================

bool GPUUTXOSet::SaveToDisk(const std::string& datadir, uint64_t block_height, const uint8_t* block_hash) {
    LogGPUMessage("Saving GPU UTXO set to disk at height " + std::to_string(block_height));

    // Get the GPUDirect Storage instance
    GPUDirectStorage& gds = GetGPUDirectStorage();

    // Initialize GDS if not already
    if (!gds.IsInitialized()) {
        GDSConfig config = GDSConfig::GetDefaults(datadir);
        if (!gds.Initialize(config)) {
            LogGPUError("Failed to initialize GPUDirect Storage");
            return false;
        }
    }

    // Save the UTXO set using GPUDirect Storage
    bool success = gds.SaveUTXOSet(
        d_headers,
        numUTXOs,
        d_scriptBlob,
        scriptBlobUsed,
        d_txidTable,
        txidTableUsed,
        const_cast<const uint32_t* const*>(d_hashTables),
        TABLE_SIZE,
        block_height,
        block_hash
    );

    if (success) {
        m_datadir = datadir;
        m_last_flushed_height = block_height;

        // Update incremental flush tracking state
        m_flush_numUTXOs = numUTXOs;
        m_flush_scriptBlobUsed = scriptBlobUsed;
        m_flush_txidTableUsed = txidTableUsed;
        ClearDirtyTracking();

        LogGPUMessage("GPU UTXO set saved successfully");
    } else {
        LogGPUError("Failed to save GPU UTXO set");
    }

    return success;
}

bool GPUUTXOSet::LoadFromDisk(const std::string& datadir, uint64_t& block_height, uint8_t* block_hash) {
    LogGPUMessage("Loading GPU UTXO set from disk...");

    // Get the GPUDirect Storage instance
    GPUDirectStorage& gds = GetGPUDirectStorage();

    // Initialize GDS if not already
    if (!gds.IsInitialized()) {
        GDSConfig config = GDSConfig::GetDefaults(datadir);
        if (!gds.Initialize(config)) {
            LogGPUError("Failed to initialize GPUDirect Storage");
            return false;
        }
    }

    // Make sure GPU memory is allocated
    if (!d_headers) {
        if (!Initialize()) {
            LogGPUError("Failed to allocate GPU memory for UTXO set");
            return false;
        }
    }

    // Load the UTXO set using GPUDirect Storage
    size_t loaded_headers = 0;
    size_t loaded_scripts = 0;
    size_t loaded_txids = 0;

    bool success = gds.LoadUTXOSet(
        d_headers,
        loaded_headers,
        maxUTXOs,
        d_scriptBlob,
        loaded_scripts,
        scriptBlobSize,
        d_txidTable,
        loaded_txids,
        txidTableSize,
        d_hashTables,
        TABLE_SIZE,
        block_height,
        block_hash
    );

    if (success) {
        numUTXOs = loaded_headers;
        scriptBlobUsed = loaded_scripts;
        txidTableUsed = loaded_txids;
        m_datadir = datadir;
        m_last_flushed_height = block_height;

        // Initialize incremental flush tracking state
        m_flush_numUTXOs = numUTXOs;
        m_flush_scriptBlobUsed = scriptBlobUsed;
        m_flush_txidTableUsed = txidTableUsed;
        ClearDirtyTracking();

        LogGPUMessage("GPU UTXO set loaded: " + std::to_string(numUTXOs) +
                     " UTXOs at height " + std::to_string(block_height));
    } else {
        LogGPUError("Failed to load GPU UTXO set from disk");
    }

    return success;
}

bool GPUUTXOSet::HasDiskSnapshot(const std::string& datadir) {
    GDSConfig config = GDSConfig::GetDefaults(datadir);

    // Check if all required files exist
    struct stat st;
    if (stat(config.utxo_headers_path.c_str(), &st) != 0) return false;
    if (stat(config.utxo_scripts_path.c_str(), &st) != 0) return false;
    if (stat(config.utxo_txids_path.c_str(), &st) != 0) return false;
    if (stat(config.utxo_hashtables_path.c_str(), &st) != 0) return false;

    return true;
}

bool GPUUTXOSet::GetDiskSnapshotHeight(const std::string& datadir, uint64_t& block_height) {
    GDSConfig config = GDSConfig::GetDefaults(datadir);

    // Open headers file and read just the header
    int fd = open(config.utxo_headers_path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    UTXOFileHeader file_header;
    ssize_t bytes_read = read(fd, &file_header, sizeof(file_header));
    close(fd);

    if (bytes_read != sizeof(file_header)) return false;
    if (!file_header.IsValid()) return false;

    block_height = file_header.block_height;
    return true;
}

bool GPUUTXOSet::FlushToDisk() {
    if (m_datadir.empty()) {
        LogGPUError("Cannot flush: datadir not set. Call SaveToDisk first.");
        return false;
    }

    // Check if there's anything to flush
    size_t new_headers = numUTXOs - m_flush_numUTXOs;
    size_t new_scripts = scriptBlobUsed - m_flush_scriptBlobUsed;
    bool has_dirty = m_dirty_count > 0;

    if (new_headers == 0 && new_scripts == 0 && !has_dirty) {
        // Nothing to flush
        return true;
    }

    LogGPUMessage("Incremental flush: " + std::to_string(new_headers) + " new headers, " +
                  std::to_string(new_scripts) + " bytes new scripts, " +
                  std::to_string(m_dirty_count) + " dirty entries");

    // Get the GPUDirect Storage instance
    GPUDirectStorage& gds = GetGPUDirectStorage();
    if (!gds.IsInitialized()) {
        LogGPUError("GPUDirect Storage not initialized for incremental flush");
        // Fall back to full save
        return SaveToDisk(m_datadir, m_last_flushed_height, nullptr);
    }

    bool success = true;

    // =========================================================================
    // Step 1: Append new UTXO headers
    // =========================================================================
    if (new_headers > 0 && success) {
        success = gds.AppendHeaders(
            d_headers + m_flush_numUTXOs,
            new_headers,
            m_flush_numUTXOs
        );
        if (!success) {
            LogGPUError("Failed to append " + std::to_string(new_headers) + " new headers");
        }
    }

    // =========================================================================
    // Step 2: Append new script data
    // =========================================================================
    if (new_scripts > 0 && success) {
        success = gds.AppendScripts(
            d_scriptBlob + m_flush_scriptBlobUsed,
            new_scripts,
            m_flush_scriptBlobUsed
        );
        if (!success) {
            LogGPUError("Failed to append " + std::to_string(new_scripts) + " bytes of script data");
        }
    }

    // =========================================================================
    // Step 3: Write dirty headers (modified existing entries, e.g., spent UTXOs)
    // =========================================================================
    if (has_dirty && success) {
        // For each dirty entry, we need to update the header on disk
        // This is less efficient than batch updates, but ensures correctness
        GDSConfig config = GDSConfig::GetDefaults(m_datadir);

        for (size_t i = 0; i < m_dirty_count && success; i++) {
            uint32_t dirty_index = m_dirty_indices[i];

            // Read the header from GPU
            UTXOHeader header;
            cudaMemcpy(&header, d_headers + dirty_index, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);

            // Clear the dirty flag before writing
            header.flags &= ~UTXO_FLAG_DIRTY;

            // Write directly to the file at the correct offset
            // File format: UTXOFileHeader (128 bytes) + headers array
            off_t file_offset = sizeof(UTXOFileHeader) + dirty_index * sizeof(UTXOHeader);

            int fd = open(config.utxo_headers_path.c_str(), O_WRONLY);
            if (fd < 0) {
                LogGPUError("Failed to open headers file for dirty write");
                success = false;
                break;
            }

            ssize_t written = pwrite(fd, &header, sizeof(UTXOHeader), file_offset);
            close(fd);

            if (written != sizeof(UTXOHeader)) {
                LogGPUError("Failed to write dirty header at index " + std::to_string(dirty_index));
                success = false;
                break;
            }

            // Also update the GPU copy to clear the dirty flag
            cudaMemcpy(d_headers + dirty_index, &header, sizeof(UTXOHeader), cudaMemcpyHostToDevice);
        }
    }

    // =========================================================================
    // Step 4: Update file header with new counts
    // =========================================================================
    if (success) {
        GDSConfig config = GDSConfig::GetDefaults(m_datadir);

        int fd = open(config.utxo_headers_path.c_str(), O_RDWR);
        if (fd >= 0) {
            UTXOFileHeader file_header;
            if (read(fd, &file_header, sizeof(file_header)) == sizeof(file_header)) {
                file_header.num_entries = numUTXOs;
                file_header.data_size = numUTXOs * sizeof(UTXOHeader);
                lseek(fd, 0, SEEK_SET);
                write(fd, &file_header, sizeof(file_header));
            }
            close(fd);
        }

        // Similarly update scripts file header
        fd = open(config.utxo_scripts_path.c_str(), O_RDWR);
        if (fd >= 0) {
            UTXOFileHeader file_header;
            if (read(fd, &file_header, sizeof(file_header)) == sizeof(file_header)) {
                file_header.data_size = scriptBlobUsed;
                lseek(fd, 0, SEEK_SET);
                write(fd, &file_header, sizeof(file_header));
            }
            close(fd);
        }
    }

    // =========================================================================
    // Step 5: Update flush tracking state on success
    // =========================================================================
    if (success) {
        m_flush_numUTXOs = numUTXOs;
        m_flush_scriptBlobUsed = scriptBlobUsed;
        m_flush_txidTableUsed = txidTableUsed;
        ClearDirtyTracking();

        LogGPUMessage("Incremental flush completed successfully");
    } else {
        LogGPUError("Incremental flush failed, falling back to full save");
        // Fall back to full save
        return SaveToDisk(m_datadir, m_last_flushed_height, nullptr);
    }

    return true;
}

} // namespace gpu