// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "gpu_direct_storage.h"
#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_logging.h"
#include "gpu_hash.cuh"

#include <cuda_runtime.h>

// Conditionally include cuFile if available
#ifdef HAVE_CUFILE
#include <cufile.h>
#else
// Stub definitions for when cuFile is not available
typedef int CUfileOpError;
typedef struct { CUfileOpError err; int cu_err; } CUfileError_t;
typedef void* CUfileHandle_t;
typedef struct { int type; union { int fd; void* handle; } handle; void* fs_ops; } CUfileDescr_t;
typedef struct { struct { unsigned int major_version; unsigned int minor_version; size_t poll_thresh_size; size_t max_direct_io_size; unsigned int dstatusflags; unsigned int dcontrolflags; } nvfs; unsigned int fflags; unsigned int max_device_cache_size; unsigned int per_buffer_cache_size; unsigned int max_device_pinned_mem_size; unsigned int max_batch_io_size; unsigned int max_batch_io_timeout_msecs; } CUfileDrvProps_t;

#define CU_FILE_SUCCESS 0
#define CU_FILE_DRIVER_NOT_INITIALIZED 5001
#define CU_FILE_HANDLE_TYPE_OPAQUE_FD 1
#define CU_FILE_NVME_SUPPORTED 4
#define CU_FILE_NVMEOF_SUPPORTED 5
#define CU_FILE_LUSTRE_SUPPORTED 0
#define CU_FILE_WEKAFS_SUPPORTED 1
#define IS_CUFILE_ERR(x) false
#define CUFILE_ERRSTR(x) "cuFile not available"

typedef void* CUstream;

static inline CUfileError_t cuFileDriverOpen() { CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, 0}; return e; }
static inline CUfileError_t cuFileDriverClose() { CUfileError_t e = {CU_FILE_SUCCESS, 0}; return e; }
static inline CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t* props) { CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, 0}; return e; }
static inline CUfileError_t cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) { CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, 0}; return e; }
static inline void cuFileHandleDeregister(CUfileHandle_t fh) {}
static inline CUfileError_t cuFileBufRegister(const void* ptr, size_t size, int flags) { CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, 0}; return e; }
static inline CUfileError_t cuFileBufDeregister(const void* ptr) { CUfileError_t e = {CU_FILE_SUCCESS, 0}; return e; }
static inline ssize_t cuFileRead(CUfileHandle_t fh, void* buf, size_t size, off_t file_offset, off_t buf_offset) { return -1; }
static inline ssize_t cuFileWrite(CUfileHandle_t fh, const void* buf, size_t size, off_t file_offset, off_t buf_offset) { return -1; }
static inline const char* cufileop_status_error(CUfileOpError err) { return "cuFile not available"; }

// Async API stubs (unused but required for API compatibility)
[[maybe_unused]] static inline CUfileError_t cuFileReadAsync(CUfileHandle_t fh, void* buf, size_t* size_p, off_t* file_offset_p, off_t* buf_offset_p, ssize_t* bytes_read_p, CUstream stream) { (void)fh; (void)buf; (void)size_p; (void)file_offset_p; (void)buf_offset_p; (void)bytes_read_p; (void)stream; CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, 0}; return e; }
[[maybe_unused]] static inline CUfileError_t cuFileWriteAsync(CUfileHandle_t fh, void* buf, size_t* size_p, off_t* file_offset_p, off_t* buf_offset_p, ssize_t* bytes_written_p, CUstream stream) { (void)fh; (void)buf; (void)size_p; (void)file_offset_p; (void)buf_offset_p; (void)bytes_written_p; (void)stream; CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, 0}; return e; }
[[maybe_unused]] static inline CUfileError_t cuFileStreamRegister(CUstream stream, unsigned int flags) { (void)stream; (void)flags; CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, 0}; return e; }
[[maybe_unused]] static inline CUfileError_t cuFileStreamDeregister(CUstream stream) { (void)stream; CUfileError_t e = {CU_FILE_SUCCESS, 0}; return e; }
#endif

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <map>

namespace gpu {

// =============================================================================
// Logging Helpers
// =============================================================================

static void LogGDS(const std::string& msg) {
    ::LogGPUInfo(("[GDS] " + msg).c_str());
}

static void LogGDSError(const std::string& msg) {
    ::LogGPUInfo(("[GDS ERROR] " + msg).c_str());
}

static void LogGDSDebug(const std::string& msg) {
    ::LogGPUDebug(("[GDS DEBUG] " + msg).c_str());
}

// =============================================================================
// GDSConfig Implementation
// =============================================================================

GDSConfig GDSConfig::GetDefaults(const std::string& datadir) {
    GDSConfig config;
    config.utxo_headers_path = datadir + "/gpu_utxo_headers.dat";
    config.utxo_scripts_path = datadir + "/gpu_utxo_scripts.dat";
    config.utxo_txids_path = datadir + "/gpu_utxo_txids.dat";
    config.utxo_hashtables_path = datadir + "/gpu_utxo_hashtables.dat";
    return config;
}

// =============================================================================
// GPUDirectStorage Implementation
// =============================================================================

// Global singleton
static std::unique_ptr<GPUDirectStorage> g_gds_instance;

GPUDirectStorage& GetGPUDirectStorage() {
    if (!g_gds_instance) {
        g_gds_instance = std::make_unique<GPUDirectStorage>();
    }
    return *g_gds_instance;
}

bool InitializeGPUDirectStorage(const GDSConfig& config) {
    return GetGPUDirectStorage().Initialize(config);
}

void ShutdownGPUDirectStorage() {
    if (g_gds_instance) {
        g_gds_instance->Shutdown();
        g_gds_instance.reset();
    }
}

GPUDirectStorage::GPUDirectStorage() = default;

GPUDirectStorage::~GPUDirectStorage() {
    Shutdown();
}

bool GPUDirectStorage::Initialize(const GDSConfig& config) {
    if (m_initialized) {
        LogGDSError("Already initialized");
        return false;
    }

    m_config = config;
    LogGDS("Initializing GPUDirect Storage...");

    // Try to initialize cuFile driver
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        if (status.err == CU_FILE_DRIVER_NOT_INITIALIZED) {
            LogGDS("GDS driver not available - using fallback mode");
            LogGDS("Tip: Install nvidia-fs driver for direct GPU-to-disk I/O");
            m_gds_available = false;
        } else {
            LogGDSError("cuFileDriverOpen failed: " + std::string(cufileop_status_error(status.err)));
            m_gds_available = false;
        }
    } else {
        // Get and log driver properties
        CUfileDrvProps_t props;
        if (cuFileDriverGetProperties(&props).err == CU_FILE_SUCCESS) {
            LogGDS("GDS driver version: " + std::to_string(props.nvfs.major_version) + "." +
                   std::to_string(props.nvfs.minor_version));
            LogGDS("Max direct I/O size: " + std::to_string(props.nvfs.max_direct_io_size / 1024) + " KB");
            LogGDS("Poll threshold: " + std::to_string(props.nvfs.poll_thresh_size / 1024) + " KB");

            // Check supported filesystems
            std::string supported = "Supported filesystems: ";
            if (props.nvfs.dstatusflags & (1 << CU_FILE_NVME_SUPPORTED)) supported += "NVMe ";
            if (props.nvfs.dstatusflags & (1 << CU_FILE_NVMEOF_SUPPORTED)) supported += "NVMeOF ";
            if (props.nvfs.dstatusflags & (1 << CU_FILE_LUSTRE_SUPPORTED)) supported += "Lustre ";
            if (props.nvfs.dstatusflags & (1 << CU_FILE_WEKAFS_SUPPORTED)) supported += "WekaFS ";
            LogGDS(supported);
        }

        m_gds_available = true;
        LogGDS("GPUDirect Storage initialized successfully");
    }

    m_initialized = true;
    return true;
}

void GPUDirectStorage::Shutdown() {
    if (!m_initialized) return;

    LogGDS("Shutting down GPUDirect Storage...");

    // Close and deregister all files
    for (auto& pair : m_file_handles) {
        CloseAndDeregisterFile(pair.first);
    }
    m_file_handles.clear();

    // Free staging buffer
    m_staging_buffer.reset();
    m_staging_buffer_size = 0;

    // Close cuFile driver
    if (m_gds_available) {
        cuFileDriverClose();
        m_gds_available = false;
    }

    m_initialized = false;
    LogGDS("GPUDirect Storage shutdown complete");
}

bool GPUDirectStorage::RegisterBuffer(void* d_buffer, size_t size) {
    if (!m_gds_available) return true;  // No-op in fallback mode

    CUfileError_t status = cuFileBufRegister(d_buffer, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        LogGDSError("cuFileBufRegister failed: " + std::string(cufileop_status_error(status.err)));
        return false;
    }
    return true;
}

void GPUDirectStorage::DeregisterBuffer(void* d_buffer) {
    if (!m_gds_available) return;
    cuFileBufDeregister(d_buffer);
}

bool GPUDirectStorage::OpenAndRegisterFile(const std::string& filepath, bool create, bool truncate) {
    // Check if already registered
    auto it = m_file_handles.find(filepath);
    if (it != m_file_handles.end() && it->second.registered) {
        return true;
    }

    // Open file with O_DIRECT for GPUDirect Storage
    int flags = O_RDWR | O_DIRECT;
    if (create) flags |= O_CREAT;
    if (truncate) flags |= O_TRUNC;

    int fd = open(filepath.c_str(), flags, 0644);
    if (fd < 0) {
        LogGDSError("Failed to open file: " + filepath + " - " + strerror(errno));
        return false;
    }

    FileHandle handle;
    handle.fd = fd;
    handle.registered = false;

    if (m_gds_available) {
        // Register with cuFile
        CUfileDescr_t descr;
        memset(&descr, 0, sizeof(descr));
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descr.handle.fd = fd;

        CUfileHandle_t cufile_handle;
        CUfileError_t status = cuFileHandleRegister(&cufile_handle, &descr);
        if (status.err != CU_FILE_SUCCESS) {
            LogGDSError("cuFileHandleRegister failed for " + filepath + ": " +
                       std::string(cufileop_status_error(status.err)));
            // Continue in fallback mode for this file
        } else {
            handle.cufile_handle = cufile_handle;
            handle.registered = true;
            LogGDSDebug("Registered file with GDS: " + filepath);
        }
    }

    m_file_handles[filepath] = handle;
    return true;
}

void GPUDirectStorage::CloseAndDeregisterFile(const std::string& filepath) {
    auto it = m_file_handles.find(filepath);
    if (it == m_file_handles.end()) return;

    FileHandle& handle = it->second;

    if (handle.registered && handle.cufile_handle) {
        cuFileHandleDeregister(static_cast<CUfileHandle_t>(handle.cufile_handle));
    }

    if (handle.fd >= 0) {
        close(handle.fd);
    }

    m_file_handles.erase(it);
}

bool GPUDirectStorage::EnsureStagingBuffer(size_t size) {
    if (m_staging_buffer_size >= size) return true;

    // Round up to 4KB alignment for O_DIRECT
    size = AlignUp(size, 4096);

    // Allocate page-aligned buffer for O_DIRECT
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size) != 0) {
        LogGDSError("Failed to allocate staging buffer: " + std::to_string(size) + " bytes");
        return false;
    }

    m_staging_buffer.reset(static_cast<uint8_t*>(ptr));
    m_staging_buffer_size = size;
    return true;
}

// =============================================================================
// Direct GPU Memory <-> Disk Operations
// =============================================================================

ssize_t GPUDirectStorage::ReadToGPU(
    const std::string& filepath,
    void* d_buffer,
    size_t size,
    off_t file_offset,
    off_t buffer_offset)
{
    if (!m_initialized) {
        LogGDSError("Not initialized");
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Open/register file if needed
    if (!OpenAndRegisterFile(filepath, false, false)) {
        return -1;
    }

    auto it = m_file_handles.find(filepath);
    if (it == m_file_handles.end()) return -1;

    ssize_t bytes_read;

    if (it->second.registered && m_gds_available) {
        // Use GPUDirect Storage - direct GPU-to-disk transfer
        bytes_read = cuFileRead(
            static_cast<CUfileHandle_t>(it->second.cufile_handle),
            d_buffer,
            size,
            file_offset,
            buffer_offset
        );

        if (bytes_read < 0) {
            if (IS_CUFILE_ERR(bytes_read)) {
                LogGDSError("cuFileRead failed: " + std::string(CUFILE_ERRSTR(bytes_read)));
            } else {
                LogGDSError("cuFileRead failed: " + std::string(strerror(-bytes_read)));
            }
            return -1;
        }

        m_stats.gds_read_ops++;
    } else {
        // Fallback to CPU-mediated transfer
        bytes_read = ReadToGPUFallback(filepath, d_buffer, size, file_offset);
        if (bytes_read < 0) return -1;
        m_stats.fallback_read_ops++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    m_stats.bytes_read += bytes_read;
    m_stats.read_ops++;
    m_stats.total_read_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

    return bytes_read;
}

ssize_t GPUDirectStorage::WriteFromGPU(
    const std::string& filepath,
    const void* d_buffer,
    size_t size,
    off_t file_offset,
    off_t buffer_offset)
{
    if (!m_initialized) {
        LogGDSError("Not initialized");
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Open/register file, create if needed
    if (!OpenAndRegisterFile(filepath, true, false)) {
        return -1;
    }

    auto it = m_file_handles.find(filepath);
    if (it == m_file_handles.end()) return -1;

    ssize_t bytes_written;

    if (it->second.registered && m_gds_available) {
        // Use GPUDirect Storage - direct GPU-to-disk transfer
        bytes_written = cuFileWrite(
            static_cast<CUfileHandle_t>(it->second.cufile_handle),
            d_buffer,
            size,
            file_offset,
            buffer_offset
        );

        if (bytes_written < 0) {
            if (IS_CUFILE_ERR(bytes_written)) {
                LogGDSError("cuFileWrite failed: " + std::string(CUFILE_ERRSTR(bytes_written)));
            } else {
                LogGDSError("cuFileWrite failed: " + std::string(strerror(-bytes_written)));
            }
            return -1;
        }

        m_stats.gds_write_ops++;
    } else {
        // Fallback to CPU-mediated transfer
        bytes_written = WriteFromGPUFallback(filepath, d_buffer, size, file_offset);
        if (bytes_written < 0) return -1;
        m_stats.fallback_write_ops++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    m_stats.bytes_written += bytes_written;
    m_stats.write_ops++;
    m_stats.total_write_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

    return bytes_written;
}

// =============================================================================
// Fallback Mode (CPU-mediated I/O)
// =============================================================================

ssize_t GPUDirectStorage::ReadToGPUFallback(
    const std::string& filepath,
    void* d_buffer,
    size_t size,
    off_t file_offset)
{
    // Ensure staging buffer is large enough
    if (!EnsureStagingBuffer(size)) {
        return -1;
    }

    auto it = m_file_handles.find(filepath);
    if (it == m_file_handles.end()) {
        // Try to open without O_DIRECT for fallback
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            LogGDSError("Failed to open file for fallback read: " + filepath);
            return -1;
        }

        FileHandle handle;
        handle.fd = fd;
        handle.registered = false;
        m_file_handles[filepath] = handle;
        it = m_file_handles.find(filepath);
    }

    // Seek and read to CPU buffer
    if (lseek(it->second.fd, file_offset, SEEK_SET) < 0) {
        LogGDSError("lseek failed: " + std::string(strerror(errno)));
        return -1;
    }

    ssize_t bytes_read = read(it->second.fd, m_staging_buffer.get(), size);
    if (bytes_read < 0) {
        LogGDSError("read failed: " + std::string(strerror(errno)));
        return -1;
    }

    // Copy from CPU to GPU
    cudaError_t err = cudaMemcpy(d_buffer, m_staging_buffer.get(), bytes_read, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LogGDSError("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        return -1;
    }

    return bytes_read;
}

ssize_t GPUDirectStorage::WriteFromGPUFallback(
    const std::string& filepath,
    const void* d_buffer,
    size_t size,
    off_t file_offset)
{
    // Ensure staging buffer is large enough
    if (!EnsureStagingBuffer(size)) {
        return -1;
    }

    // Copy from GPU to CPU
    cudaError_t err = cudaMemcpy(m_staging_buffer.get(), d_buffer, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LogGDSError("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        return -1;
    }

    auto it = m_file_handles.find(filepath);
    if (it == m_file_handles.end()) {
        // Open without O_DIRECT for fallback
        int fd = open(filepath.c_str(), O_WRONLY | O_CREAT, 0644);
        if (fd < 0) {
            LogGDSError("Failed to open file for fallback write: " + filepath);
            return -1;
        }

        FileHandle handle;
        handle.fd = fd;
        handle.registered = false;
        m_file_handles[filepath] = handle;
        it = m_file_handles.find(filepath);
    }

    // Seek and write from CPU buffer
    if (lseek(it->second.fd, file_offset, SEEK_SET) < 0) {
        LogGDSError("lseek failed: " + std::string(strerror(errno)));
        return -1;
    }

    ssize_t bytes_written = write(it->second.fd, m_staging_buffer.get(), size);
    if (bytes_written < 0) {
        LogGDSError("write failed: " + std::string(strerror(errno)));
        return -1;
    }

    // Ensure data is on disk
    fsync(it->second.fd);

    return bytes_written;
}

// =============================================================================
// UTXO Set Persistence Operations
// =============================================================================

bool GPUDirectStorage::SaveUTXOSet(
    const UTXOHeader* d_headers,
    size_t num_headers,
    const uint8_t* d_scripts,
    size_t scripts_size,
    const uint256_gpu* d_txids,
    size_t num_txids,
    const uint32_t* const* d_hash_tables,
    size_t table_size,
    uint64_t block_height,
    const uint8_t* block_hash)
{
    LogGDS("Saving GPU UTXO set: " + std::to_string(num_headers) + " headers, " +
           std::to_string(scripts_size) + " bytes scripts, " +
           std::to_string(num_txids) + " txids");

    auto start = std::chrono::high_resolution_clock::now();

    // Prepare file header
    UTXOFileHeader file_header;
    memset(&file_header, 0, sizeof(file_header));
    file_header.magic = UTXOFileHeader::MAGIC;
    file_header.version = UTXOFileHeader::CURRENT_VERSION;
    file_header.block_height = block_height;
    if (block_hash) {
        memcpy(file_header.block_hash, block_hash, 32);
    }

    // Save headers
    {
        file_header.num_entries = num_headers;
        file_header.data_size = num_headers * sizeof(UTXOHeader);

        // Open file for writing, truncate existing
        if (!OpenAndRegisterFile(m_config.utxo_headers_path, true, true)) {
            return false;
        }

        // Write file header (use fallback since it's small and host-side)
        if (!EnsureStagingBuffer(sizeof(file_header))) return false;
        memcpy(m_staging_buffer.get(), &file_header, sizeof(file_header));

        auto it = m_file_handles.find(m_config.utxo_headers_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            write(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to write headers file header");
            return false;
        }

        // Write header data directly from GPU
        ssize_t written = WriteFromGPU(
            m_config.utxo_headers_path,
            d_headers,
            file_header.data_size,
            sizeof(UTXOFileHeader),
            0
        );
        if (written != static_cast<ssize_t>(file_header.data_size)) {
            LogGDSError("Failed to write UTXO headers");
            return false;
        }
    }

    // Save scripts
    {
        file_header.num_entries = 1;  // Single blob
        file_header.data_size = scripts_size;

        if (!OpenAndRegisterFile(m_config.utxo_scripts_path, true, true)) {
            return false;
        }

        // Write file header
        memcpy(m_staging_buffer.get(), &file_header, sizeof(file_header));
        auto it = m_file_handles.find(m_config.utxo_scripts_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            write(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to write scripts file header");
            return false;
        }

        // Write script blob directly from GPU
        ssize_t written = WriteFromGPU(
            m_config.utxo_scripts_path,
            d_scripts,
            scripts_size,
            sizeof(UTXOFileHeader),
            0
        );
        if (written != static_cast<ssize_t>(scripts_size)) {
            LogGDSError("Failed to write script blob");
            return false;
        }
    }

    // Save txid table
    {
        file_header.num_entries = num_txids;
        file_header.data_size = num_txids * sizeof(uint256_gpu);

        if (!OpenAndRegisterFile(m_config.utxo_txids_path, true, true)) {
            return false;
        }

        memcpy(m_staging_buffer.get(), &file_header, sizeof(file_header));
        auto it = m_file_handles.find(m_config.utxo_txids_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            write(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to write txids file header");
            return false;
        }

        ssize_t written = WriteFromGPU(
            m_config.utxo_txids_path,
            d_txids,
            file_header.data_size,
            sizeof(UTXOFileHeader),
            0
        );
        if (written != static_cast<ssize_t>(file_header.data_size)) {
            LogGDSError("Failed to write txid table");
            return false;
        }
    }

    // Save hash tables (4 tables concatenated)
    {
        size_t table_bytes = table_size * sizeof(uint32_t);
        file_header.num_entries = 4;  // 4 hash tables
        file_header.data_size = 4 * table_bytes;

        if (!OpenAndRegisterFile(m_config.utxo_hashtables_path, true, true)) {
            return false;
        }

        memcpy(m_staging_buffer.get(), &file_header, sizeof(file_header));
        auto it = m_file_handles.find(m_config.utxo_hashtables_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            write(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to write hash tables file header");
            return false;
        }

        // Write each hash table
        for (int i = 0; i < 4; i++) {
            ssize_t written = WriteFromGPU(
                m_config.utxo_hashtables_path,
                d_hash_tables[i],
                table_bytes,
                sizeof(UTXOFileHeader) + i * table_bytes,
                0
            );
            if (written != static_cast<ssize_t>(table_bytes)) {
                LogGDSError("Failed to write hash table " + std::to_string(i));
                return false;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    size_t total_bytes = num_headers * sizeof(UTXOHeader) + scripts_size +
                        num_txids * sizeof(uint256_gpu) + 4 * table_size * sizeof(uint32_t);
    double throughput_mbps = (total_bytes / 1024.0 / 1024.0) / (elapsed_ms / 1000.0);

    LogGDS("UTXO set saved in " + std::to_string(elapsed_ms) + " ms (" +
           std::to_string(throughput_mbps) + " MB/s)");

    return true;
}

bool GPUDirectStorage::LoadUTXOSet(
    UTXOHeader* d_headers,
    size_t& num_headers,
    size_t max_headers,
    uint8_t* d_scripts,
    size_t& scripts_size,
    size_t max_scripts,
    uint256_gpu* d_txids,
    size_t& num_txids,
    size_t max_txids,
    uint32_t** d_hash_tables,
    size_t table_size,
    uint64_t& block_height,
    uint8_t* block_hash)
{
    LogGDS("Loading GPU UTXO set from disk...");

    auto start = std::chrono::high_resolution_clock::now();

    UTXOFileHeader file_header;

    // Load headers
    {
        if (!OpenAndRegisterFile(m_config.utxo_headers_path, false, false)) {
            LogGDSError("Headers file not found: " + m_config.utxo_headers_path);
            return false;
        }

        // Read file header
        if (!EnsureStagingBuffer(sizeof(file_header))) return false;
        auto it = m_file_handles.find(m_config.utxo_headers_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            read(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to read headers file header");
            return false;
        }
        memcpy(&file_header, m_staging_buffer.get(), sizeof(file_header));

        if (!file_header.IsValid()) {
            LogGDSError("Invalid headers file format");
            return false;
        }

        num_headers = std::min(static_cast<size_t>(file_header.num_entries), max_headers);
        block_height = file_header.block_height;
        if (block_hash) {
            memcpy(block_hash, file_header.block_hash, 32);
        }

        // Read header data directly to GPU
        ssize_t bytes_read = ReadToGPU(
            m_config.utxo_headers_path,
            d_headers,
            num_headers * sizeof(UTXOHeader),
            sizeof(UTXOFileHeader),
            0
        );
        if (bytes_read != static_cast<ssize_t>(num_headers * sizeof(UTXOHeader))) {
            LogGDSError("Failed to read UTXO headers");
            return false;
        }
    }

    // Load scripts
    {
        if (!OpenAndRegisterFile(m_config.utxo_scripts_path, false, false)) {
            LogGDSError("Scripts file not found");
            return false;
        }

        auto it = m_file_handles.find(m_config.utxo_scripts_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            read(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to read scripts file header");
            return false;
        }
        memcpy(&file_header, m_staging_buffer.get(), sizeof(file_header));

        scripts_size = std::min(static_cast<size_t>(file_header.data_size), max_scripts);

        ssize_t bytes_read = ReadToGPU(
            m_config.utxo_scripts_path,
            d_scripts,
            scripts_size,
            sizeof(UTXOFileHeader),
            0
        );
        if (bytes_read != static_cast<ssize_t>(scripts_size)) {
            LogGDSError("Failed to read script blob");
            return false;
        }
    }

    // Load txid table
    {
        if (!OpenAndRegisterFile(m_config.utxo_txids_path, false, false)) {
            LogGDSError("Txids file not found");
            return false;
        }

        auto it = m_file_handles.find(m_config.utxo_txids_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            read(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to read txids file header");
            return false;
        }
        memcpy(&file_header, m_staging_buffer.get(), sizeof(file_header));

        num_txids = std::min(static_cast<size_t>(file_header.num_entries), max_txids);

        ssize_t bytes_read = ReadToGPU(
            m_config.utxo_txids_path,
            d_txids,
            num_txids * sizeof(uint256_gpu),
            sizeof(UTXOFileHeader),
            0
        );
        if (bytes_read != static_cast<ssize_t>(num_txids * sizeof(uint256_gpu))) {
            LogGDSError("Failed to read txid table");
            return false;
        }
    }

    // Load hash tables
    {
        if (!OpenAndRegisterFile(m_config.utxo_hashtables_path, false, false)) {
            LogGDSError("Hash tables file not found");
            return false;
        }

        auto it = m_file_handles.find(m_config.utxo_hashtables_path);
        if (lseek(it->second.fd, 0, SEEK_SET) < 0 ||
            read(it->second.fd, m_staging_buffer.get(), sizeof(file_header)) != sizeof(file_header)) {
            LogGDSError("Failed to read hash tables file header");
            return false;
        }
        memcpy(&file_header, m_staging_buffer.get(), sizeof(file_header));

        size_t table_bytes = table_size * sizeof(uint32_t);

        for (int i = 0; i < 4; i++) {
            ssize_t bytes_read = ReadToGPU(
                m_config.utxo_hashtables_path,
                d_hash_tables[i],
                table_bytes,
                sizeof(UTXOFileHeader) + i * table_bytes,
                0
            );
            if (bytes_read != static_cast<ssize_t>(table_bytes)) {
                LogGDSError("Failed to read hash table " + std::to_string(i));
                return false;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    size_t total_bytes = num_headers * sizeof(UTXOHeader) + scripts_size +
                        num_txids * sizeof(uint256_gpu) + 4 * table_size * sizeof(uint32_t);
    double throughput_mbps = (total_bytes / 1024.0 / 1024.0) / (elapsed_ms / 1000.0);

    LogGDS("UTXO set loaded: " + std::to_string(num_headers) + " headers, " +
           std::to_string(scripts_size) + " bytes scripts, block height " +
           std::to_string(block_height));
    LogGDS("Load time: " + std::to_string(elapsed_ms) + " ms (" +
           std::to_string(throughput_mbps) + " MB/s)");

    return true;
}

// =============================================================================
// Incremental Update Operations
// =============================================================================

bool GPUDirectStorage::AppendHeaders(
    const UTXOHeader* d_headers,
    size_t num_headers,
    size_t starting_index)
{
    if (!OpenAndRegisterFile(m_config.utxo_headers_path, true, false)) {
        return false;
    }

    off_t file_offset = sizeof(UTXOFileHeader) + starting_index * sizeof(UTXOHeader);
    ssize_t written = WriteFromGPU(
        m_config.utxo_headers_path,
        d_headers,
        num_headers * sizeof(UTXOHeader),
        file_offset,
        0
    );

    return written == static_cast<ssize_t>(num_headers * sizeof(UTXOHeader));
}

bool GPUDirectStorage::AppendScripts(
    const uint8_t* d_scripts,
    size_t size,
    size_t starting_offset)
{
    if (!OpenAndRegisterFile(m_config.utxo_scripts_path, true, false)) {
        return false;
    }

    off_t file_offset = sizeof(UTXOFileHeader) + starting_offset;
    ssize_t written = WriteFromGPU(
        m_config.utxo_scripts_path,
        d_scripts,
        size,
        file_offset,
        0
    );

    return written == static_cast<ssize_t>(size);
}

bool GPUDirectStorage::UpdateHashTable(
    int table_index,
    size_t slot_index,
    uint32_t new_value)
{
    if (table_index < 0 || table_index >= 4) return false;

    if (!OpenAndRegisterFile(m_config.utxo_hashtables_path, true, false)) {
        return false;
    }

    size_t table_bytes = GPUUTXOSet::TABLE_SIZE * sizeof(uint32_t);
    off_t file_offset = sizeof(UTXOFileHeader) +
                       table_index * table_bytes +
                       slot_index * sizeof(uint32_t);

    // For single uint32_t updates, use fallback (not worth GDS overhead)
    if (!EnsureStagingBuffer(sizeof(uint32_t))) return false;
    memcpy(m_staging_buffer.get(), &new_value, sizeof(uint32_t));

    auto it = m_file_handles.find(m_config.utxo_hashtables_path);
    if (it == m_file_handles.end()) return false;

    if (lseek(it->second.fd, file_offset, SEEK_SET) < 0) return false;
    return write(it->second.fd, m_staging_buffer.get(), sizeof(uint32_t)) == sizeof(uint32_t);
}

// =============================================================================
// Async Operations with CUDA Streams
// =============================================================================

bool GPUDirectStorage::RegisterStream(void* stream) {
    if (!m_gds_available || !stream) return false;

    // Check if already registered
    if (m_registered_streams.find(stream) != m_registered_streams.end()) {
        return true;
    }

#ifdef HAVE_CUFILE
    CUfileError_t status = cuFileStreamRegister(
        static_cast<CUstream>(stream),
        CU_FILE_STREAM_FIXED_BUF_OFFSET | CU_FILE_STREAM_FIXED_FILE_OFFSET
    );
    if (status.err != CU_FILE_SUCCESS) {
        LogGDSError("cuFileStreamRegister failed: " + std::string(cufileop_status_error(status.err)));
        return false;
    }
    m_registered_streams.insert(stream);
    LogGDSDebug("Registered CUDA stream for async I/O");
#endif
    return true;
}

void GPUDirectStorage::DeregisterStream(void* stream) {
    if (!stream) return;

    auto it = m_registered_streams.find(stream);
    if (it == m_registered_streams.end()) return;

#ifdef HAVE_CUFILE
    cuFileStreamDeregister(static_cast<CUstream>(stream));
#endif
    m_registered_streams.erase(it);
}

bool GPUDirectStorage::ReadToGPUAsync(
    const std::string& filepath,
    void* d_buffer,
    size_t size,
    off_t file_offset,
    void* stream)
{
    if (!m_initialized) {
        LogGDSError("Not initialized");
        return false;
    }

    // Open/register file if needed
    if (!OpenAndRegisterFile(filepath, false, false)) {
        return false;
    }

    auto it = m_file_handles.find(filepath);
    if (it == m_file_handles.end()) return false;

#ifdef HAVE_CUFILE
    if (it->second.registered && m_gds_available && stream) {
        // Register stream if not already registered
        RegisterStream(stream);

        // Create async operation tracking
        AsyncOp op;
        op.stream = stream;
        op.size = size;
        op.file_offset = file_offset;
        op.buffer_offset = 0;
        op.bytes_transferred = 0;
        op.is_read = true;
        op.completed = false;

        // Use cuFileReadAsync for true async I/O
        size_t size_copy = size;
        off_t file_offset_copy = file_offset;
        off_t buffer_offset_copy = 0;

        CUfileError_t status = cuFileReadAsync(
            static_cast<CUfileHandle_t>(it->second.cufile_handle),
            d_buffer,
            &size_copy,
            &file_offset_copy,
            &buffer_offset_copy,
            &op.bytes_transferred,
            static_cast<CUstream>(stream)
        );

        if (status.err != CU_FILE_SUCCESS) {
            LogGDSError("cuFileReadAsync failed: " + std::string(cufileop_status_error(status.err)));
            // Fall back to synchronous
            ssize_t result = ReadToGPU(filepath, d_buffer, size, file_offset, 0);
            return result >= 0;
        }

        m_pending_async_ops.push_back(op);
        m_stats.gds_read_ops++;
        return true;
    }
#endif

    // Fallback to synchronous read
    ssize_t result = ReadToGPU(filepath, d_buffer, size, file_offset, 0);
    return result >= 0;
}

bool GPUDirectStorage::WriteFromGPUAsync(
    const std::string& filepath,
    const void* d_buffer,
    size_t size,
    off_t file_offset,
    void* stream)
{
    if (!m_initialized) {
        LogGDSError("Not initialized");
        return false;
    }

    // Open/register file, create if needed
    if (!OpenAndRegisterFile(filepath, true, false)) {
        return false;
    }

    auto it = m_file_handles.find(filepath);
    if (it == m_file_handles.end()) return false;

#ifdef HAVE_CUFILE
    if (it->second.registered && m_gds_available && stream) {
        // Register stream if not already registered
        RegisterStream(stream);

        // Create async operation tracking
        AsyncOp op;
        op.stream = stream;
        op.size = size;
        op.file_offset = file_offset;
        op.buffer_offset = 0;
        op.bytes_transferred = 0;
        op.is_read = false;
        op.completed = false;

        // Use cuFileWriteAsync for true async I/O
        size_t size_copy = size;
        off_t file_offset_copy = file_offset;
        off_t buffer_offset_copy = 0;

        CUfileError_t status = cuFileWriteAsync(
            static_cast<CUfileHandle_t>(it->second.cufile_handle),
            const_cast<void*>(d_buffer),
            &size_copy,
            &file_offset_copy,
            &buffer_offset_copy,
            &op.bytes_transferred,
            static_cast<CUstream>(stream)
        );

        if (status.err != CU_FILE_SUCCESS) {
            LogGDSError("cuFileWriteAsync failed: " + std::string(cufileop_status_error(status.err)));
            // Fall back to synchronous
            ssize_t result = WriteFromGPU(filepath, d_buffer, size, file_offset, 0);
            return result >= 0;
        }

        m_pending_async_ops.push_back(op);
        m_stats.gds_write_ops++;
        return true;
    }
#endif

    // Fallback to synchronous write
    ssize_t result = WriteFromGPU(filepath, d_buffer, size, file_offset, 0);
    return result >= 0;
}

bool GPUDirectStorage::SyncAll() {
    // Sync all CUDA streams with pending async ops
    for (auto& op : m_pending_async_ops) {
        if (op.stream && !op.completed) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(op.stream));
            op.completed = true;
        }
    }
    m_pending_async_ops.clear();

    // Sync all open files to disk
    for (auto& pair : m_file_handles) {
        if (pair.second.fd >= 0) {
            fsync(pair.second.fd);
        }
    }
    return true;
}

void GPUDirectStorage::ResetStats() {
    m_stats = Stats{};
}

} // namespace gpu
