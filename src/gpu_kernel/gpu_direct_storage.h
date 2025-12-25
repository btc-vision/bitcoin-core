// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_DIRECT_STORAGE_H
#define BITCOIN_GPU_KERNEL_GPU_DIRECT_STORAGE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <vector>

// Forward declarations
namespace gpu {
    struct UTXOHeader;
    struct uint256_gpu;
}

namespace gpu {

// =============================================================================
// GPUDirect Storage Configuration
// =============================================================================

struct GDSConfig {
    // File paths for UTXO data storage
    std::string utxo_headers_path;      // Path for UTXO headers file
    std::string utxo_scripts_path;      // Path for script blob file
    std::string utxo_txids_path;        // Path for txid table file
    std::string utxo_hashtables_path;   // Path for hash tables file

    // Performance tuning
    size_t io_block_size{4096};         // Alignment for direct I/O (typically 4KB)
    size_t max_batch_size{16 * 1024 * 1024};  // Max batch size for single I/O op (16MB)
    bool use_async_io{true};            // Use async I/O with CUDA streams
    bool enable_checksums{true};        // Enable data integrity checksums

    // Default paths relative to datadir
    static GDSConfig GetDefaults(const std::string& datadir);
};

// =============================================================================
// UTXO Storage File Header (persisted at beginning of each file)
// =============================================================================

struct UTXOFileHeader {
    uint32_t magic;                     // Magic number: 0x55545830 ("UTX0") - 4 bytes
    uint32_t version;                   // File format version - 4 bytes
    uint64_t num_entries;               // Number of entries in file - 8 bytes
    uint64_t data_size;                 // Size of data section in bytes - 8 bytes
    uint64_t checksum;                  // SHA256 checksum of data (first 8 bytes) - 8 bytes
    uint64_t block_height;              // Block height when snapshot was taken - 8 bytes
    uint8_t  block_hash[32];            // Block hash when snapshot was taken - 32 bytes
    uint8_t  reserved[56];              // Reserved for future use - 56 bytes
    // Total: 4 + 4 + 8 + 8 + 8 + 8 + 32 + 56 = 128 bytes

    static constexpr uint32_t MAGIC = 0x55545830;  // "UTX0"
    static constexpr uint32_t CURRENT_VERSION = 1;

    bool IsValid() const {
        return magic == MAGIC && version <= CURRENT_VERSION;
    }
};

static_assert(sizeof(UTXOFileHeader) == 128, "UTXOFileHeader must be 128 bytes");

// =============================================================================
// GPUDirect Storage Manager
// =============================================================================

class GPUDirectStorage {
public:
    GPUDirectStorage();
    ~GPUDirectStorage();

    // Initialization
    bool Initialize(const GDSConfig& config);
    bool IsInitialized() const { return m_initialized; }
    bool IsGDSAvailable() const { return m_gds_available; }
    void Shutdown();

    // ==========================================================================
    // Direct GPU Memory <-> Disk Operations
    // These bypass CPU entirely using NVIDIA GPUDirect Storage
    // ==========================================================================

    // Read data directly from disk to GPU memory
    // Returns bytes read, or -1 on error
    ssize_t ReadToGPU(
        const std::string& filepath,
        void* d_buffer,                 // Device pointer (GPU memory)
        size_t size,
        off_t file_offset = 0,
        off_t buffer_offset = 0
    );

    // Write data directly from GPU memory to disk
    // Returns bytes written, or -1 on error
    ssize_t WriteFromGPU(
        const std::string& filepath,
        const void* d_buffer,           // Device pointer (GPU memory)
        size_t size,
        off_t file_offset = 0,
        off_t buffer_offset = 0
    );

    // ==========================================================================
    // UTXO Set Persistence Operations
    // ==========================================================================

    // Save entire GPU UTXO set to disk (creates/overwrites files)
    bool SaveUTXOSet(
        const UTXOHeader* d_headers,
        size_t num_headers,
        const uint8_t* d_scripts,
        size_t scripts_size,
        const uint256_gpu* d_txids,
        size_t num_txids,
        const uint32_t* const* d_hash_tables,  // 4 tables
        size_t table_size,
        uint64_t block_height,
        const uint8_t* block_hash
    );

    // Load entire GPU UTXO set from disk
    bool LoadUTXOSet(
        UTXOHeader* d_headers,
        size_t& num_headers,
        size_t max_headers,
        uint8_t* d_scripts,
        size_t& scripts_size,
        size_t max_scripts,
        uint256_gpu* d_txids,
        size_t& num_txids,
        size_t max_txids,
        uint32_t** d_hash_tables,          // 4 tables
        size_t table_size,
        uint64_t& block_height,
        uint8_t* block_hash
    );

    // ==========================================================================
    // Incremental Update Operations (for real-time UTXO updates)
    // ==========================================================================

    // Append new UTXO headers to the end of the file
    bool AppendHeaders(
        const UTXOHeader* d_headers,
        size_t num_headers,
        size_t starting_index           // Where in the file to write
    );

    // Append new script data to the script blob
    bool AppendScripts(
        const uint8_t* d_scripts,
        size_t size,
        size_t starting_offset          // Offset in the script blob file
    );

    // Update hash table entry (marks entry as spent or adds new)
    bool UpdateHashTable(
        int table_index,                // 0-3
        size_t slot_index,
        uint32_t new_value
    );

    // ==========================================================================
    // Async Operations with CUDA Streams
    // ==========================================================================

    // Start async read to GPU (returns immediately, use SyncStream to wait)
    bool ReadToGPUAsync(
        const std::string& filepath,
        void* d_buffer,
        size_t size,
        off_t file_offset,
        void* stream                    // cudaStream_t
    );

    // Start async write from GPU
    bool WriteFromGPUAsync(
        const std::string& filepath,
        const void* d_buffer,
        size_t size,
        off_t file_offset,
        void* stream                    // cudaStream_t
    );

    // Synchronize all pending async operations
    bool SyncAll();

    // ==========================================================================
    // Fallback Mode (when GDS is not available)
    // ==========================================================================

    // Uses standard CPU-mediated I/O when GDS is unavailable
    ssize_t ReadToGPUFallback(
        const std::string& filepath,
        void* d_buffer,
        size_t size,
        off_t file_offset
    );

    ssize_t WriteFromGPUFallback(
        const std::string& filepath,
        const void* d_buffer,
        size_t size,
        off_t file_offset
    );

    // ==========================================================================
    // Statistics and Diagnostics
    // ==========================================================================

    struct Stats {
        uint64_t bytes_read;
        uint64_t bytes_written;
        uint64_t read_ops;
        uint64_t write_ops;
        uint64_t gds_read_ops;          // Ops using actual GDS
        uint64_t gds_write_ops;
        uint64_t fallback_read_ops;     // Ops using CPU fallback
        uint64_t fallback_write_ops;
        double total_read_time_ms;
        double total_write_time_ms;
    };

    Stats GetStats() const { return m_stats; }
    void ResetStats();

private:
    bool m_initialized{false};
    bool m_gds_available{false};
    GDSConfig m_config;
    Stats m_stats{};

    // Internal buffer for fallback operations
    std::unique_ptr<uint8_t[]> m_staging_buffer;
    size_t m_staging_buffer_size{0};

    // File handles for registered files (opaque handles from cuFile)
    struct FileHandle {
        void* cufile_handle{nullptr};   // CUfileHandle_t
        int fd{-1};
        bool registered{false};
    };
    std::map<std::string, FileHandle> m_file_handles;

    // Async operation tracking
    struct AsyncOp {
        void* stream;                   // cudaStream_t
        size_t size;
        off_t file_offset;
        off_t buffer_offset;
        ssize_t bytes_transferred;
        bool is_read;                   // true=read, false=write
        bool completed;
    };
    std::vector<AsyncOp> m_pending_async_ops;
    std::set<void*> m_registered_streams;  // Streams registered with cuFileStreamRegister

    // Internal helpers
    bool RegisterBuffer(void* d_buffer, size_t size);
    void DeregisterBuffer(void* d_buffer);
    bool OpenAndRegisterFile(const std::string& filepath, bool create, bool truncate);
    void CloseAndDeregisterFile(const std::string& filepath);
    bool RegisterStream(void* stream);
    void DeregisterStream(void* stream);
    bool EnsureStagingBuffer(size_t size);

    // Alignment helpers
    size_t AlignUp(size_t value, size_t alignment) const {
        return (value + alignment - 1) & ~(alignment - 1);
    }
};

// =============================================================================
// Global accessor for the GPU Direct Storage singleton
// =============================================================================

GPUDirectStorage& GetGPUDirectStorage();
bool InitializeGPUDirectStorage(const GDSConfig& config);
void ShutdownGPUDirectStorage();

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_DIRECT_STORAGE_H
