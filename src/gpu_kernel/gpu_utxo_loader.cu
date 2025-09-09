#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_utils.h"
#include "gpu_logging.h"
#include <vector>
#include <chrono>
#include <cstring>

namespace gpu {

// Helper function for logging
static inline void LogMsg(const std::string& msg) {
    ::LogGPUDebug(msg.c_str());
}

// Forward declaration of CPU loader function
extern bool LoadUTXOSetToCPU(const void* coinsCache, 
                             std::vector<UTXOHeader>& headers,
                             std::vector<uint256_gpu>& uniqueTxids,
                             std::vector<uint8_t>& scriptBlob,
                             size_t maxUTXOs,
                             size_t maxScriptBlobSize);

// GPU-side loader that uploads data from CPU buffers
bool GPUUTXOSet::LoadFromCPU(const void* coinsCache) {
    LogMsg("Starting UTXO set load from CPU");
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (!d_headers || !d_scriptBlob || !d_txidTable) {
        LogMsg("GPU memory not initialized");
        return false;
    }
    
    // Prepare CPU-side buffers
    std::vector<UTXOHeader> headers;
    std::vector<uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    // Load data on CPU side
    if (!LoadUTXOSetToCPU(coinsCache, headers, uniqueTxids, scriptBlob, maxUTXOs, scriptBlobSize)) {
        LogMsg("Failed to load UTXO set on CPU");
        return false;
    }
    
    numUTXOs = headers.size();
    scriptBlobUsed = scriptBlob.size();
    txidTableUsed = uniqueTxids.size();
    
    LogMsg("Uploading " + std::to_string(numUTXOs) + " UTXOs to GPU");
    
    // Upload headers to GPU
    cudaMemcpy(d_headers, headers.data(), 
               headers.size() * sizeof(UTXOHeader),
               cudaMemcpyHostToDevice);
    
    // Upload script blob to GPU
    cudaMemcpy(d_scriptBlob, scriptBlob.data(),
               scriptBlob.size(),
               cudaMemcpyHostToDevice);
    
    // Upload txid table to GPU
    cudaMemcpy(d_txidTable, uniqueTxids.data(),
               uniqueTxids.size() * sizeof(uint256_gpu),
               cudaMemcpyHostToDevice);
    
    // Build hash tables on GPU
    LogMsg("Building hash tables on GPU...");
    
    // Clear hash tables first
    for (int i = 0; i < 4; i++) {
        cudaMemset(d_hashTables[i], 0xFF, TABLE_SIZE * sizeof(uint32_t));
    }
    
    // For each UTXO, insert into hash tables
    for (uint32_t i = 0; i < numUTXOs; i++) {
        // Get the original txid from our uniqueTxids table using the header's index
        uint256_gpu gpu_txid = uniqueTxids[headers[i].txid_index];
        
        // Convert to uint256 for hash functions
        uint256 txid_uint256;
        memcpy(txid_uint256.begin(), gpu_txid.data, 32);
        
        uint32_t hashes[4] = {
            Hash1(txid_uint256, headers[i].vout),
            Hash2(txid_uint256, headers[i].vout),
            Hash3(txid_uint256, headers[i].vout),
            Hash4(txid_uint256, headers[i].vout)
        };
        
        // Try to insert in first available slot
        bool inserted = false;
        for (int j = 0; j < 4; j++) {
            uint32_t empty = 0xFFFFFFFF;
            uint32_t* addr = d_hashTables[j] + hashes[j];
            cudaMemcpy(&empty, addr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            if (empty == 0xFFFFFFFF) {
                cudaMemcpy(addr, &i, sizeof(uint32_t), cudaMemcpyHostToDevice);
                inserted = true;
                break;
            }
        }
        
        if (!inserted) {
            LogMsg("Failed to insert UTXO " + std::to_string(i) + " - hash collision");
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    LogMsg("UTXO set loaded successfully in " + std::to_string(duration.count()) + " ms");
    LogMsg("Load factor: " + std::to_string(GetLoadFactor()) + "%");
    
    return true;
}

} // namespace gpu