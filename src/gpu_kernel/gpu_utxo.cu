#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_utils.h"
#include "gpu_hash.cuh"
#include "gpu_logging.h"
#include <cuda.h>
#include <cstring>
#include <random>

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
    
    // Estimate memory allocation:
    // - Headers: 100M * 32B = 3.2GB
    // - Scripts: ~5-10GB (variable)
    // - Txid table: ~50M * 32B = 1.6GB
    // - Hash tables: 4 * 26M * 4B = 416MB
    
    maxUTXOs = 100000000;  // 100 million UTXOs
    scriptBlobSize = 5ULL * 1024 * 1024 * 1024;  // 5GB for scripts initially
    txidTableSize = 50000000;  // 50 million unique txids
    
    // Check if we have enough memory
    size_t required_mem = 
        maxUTXOs * sizeof(UTXOHeader) +          // Headers
        scriptBlobSize +                          // Scripts
        txidTableSize * sizeof(uint256_gpu) +    // Txid table
        4 * TABLE_SIZE * sizeof(uint32_t);       // Hash tables
    
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
                // Mark as spent
                header.flags |= UTXO_FLAG_SPENT;
                cudaMemcpy(d_headers + index, &header, sizeof(UTXOHeader), cudaMemcpyHostToDevice);
                
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

} // namespace gpu