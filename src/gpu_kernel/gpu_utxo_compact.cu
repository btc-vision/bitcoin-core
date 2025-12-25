#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_hash.cuh"
#include "gpu_logging.h"
#include <vector>
#include <chrono>

namespace gpu {

// Helper function for logging
static inline void LogMsg(const std::string& msg) {
    ::LogGPUDebug(msg.c_str());
}

// Compact the UTXO set (full implementation)
bool GPUUTXOSet::Compact() {
    LogMsg("Starting full UTXO set compaction");
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (totalFreeSpace <= totalVRAMUsed * 0.1) {
        LogMsg("Compaction not needed - free space below 10% threshold");
        return true;
    }
    
    // Allocate new memory for compacted data
    UTXOHeader* d_newHeaders;
    uint8_t* d_newScriptBlob;
    
    cudaError_t err = cudaMalloc(&d_newHeaders, maxUTXOs * sizeof(UTXOHeader));
    if (err != cudaSuccess) {
        LogMsg("Failed to allocate new header memory for compaction");
        return false;
    }
    
    err = cudaMalloc(&d_newScriptBlob, scriptBlobSize);
    if (err != cudaSuccess) {
        cudaFree(d_newHeaders);
        LogMsg("Failed to allocate new script blob for compaction");
        return false;
    }
    
    // Copy non-spent UTXOs to host for processing
    std::vector<UTXOHeader> headers(numUTXOs);
    cudaMemcpy(headers.data(), d_headers, numUTXOs * sizeof(UTXOHeader), cudaMemcpyDeviceToHost);
    
    // Build list of active UTXOs and compact
    std::vector<UTXOHeader> compactedHeaders;
    uint64_t newScriptOffset = 0;
    uint32_t newIndex = 0;
    
    for (uint32_t i = 0; i < numUTXOs; i++) {
        if (!(headers[i].flags & UTXO_FLAG_SPENT)) {
            UTXOHeader header = headers[i];
            
            // Copy script data to new location
            cudaMemcpy(d_newScriptBlob + newScriptOffset,
                       d_scriptBlob + header.script_offset,
                       header.script_size,
                       cudaMemcpyDeviceToDevice);
            
            header.script_offset = newScriptOffset;
            newScriptOffset += header.script_size;
            
            compactedHeaders.push_back(header);
            newIndex++;
        }
    }
    
    LogMsg("Compacting " + std::to_string(compactedHeaders.size()) + " active UTXOs");
    
    // Upload compacted headers
    cudaMemcpy(d_newHeaders, compactedHeaders.data(),
               compactedHeaders.size() * sizeof(UTXOHeader),
               cudaMemcpyHostToDevice);
    
    // Clear and rebuild hash tables
    for (int i = 0; i < 4; i++) {
        cudaMemset(d_hashTables[i], 0xFF, TABLE_SIZE * sizeof(uint32_t));
    }
    
    // Rebuild hash table entries for compacted UTXOs
    for (uint32_t i = 0; i < compactedHeaders.size(); i++) {
        // Get txid for this UTXO
        uint256_gpu gpu_txid;
        cudaMemcpy(&gpu_txid, d_txidTable + compactedHeaders[i].txid_index, 
                   sizeof(uint256_gpu), cudaMemcpyDeviceToHost);
        
        // Compute hash using SipHash for proper distribution
        uint64_t siphash_result = SipHashUint256(0x0123456789abcdefULL, 0xfedcba9876543210ULL, gpu_txid);
        uint32_t h1 = siphash_result % TABLE_SIZE;
        
        // Insert into first available table
        uint32_t* addr = d_hashTables[0] + h1;
        cudaMemcpy(addr, &i, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    
    // Swap buffers
    cudaFree(d_headers);
    cudaFree(d_scriptBlob);
    
    d_headers = d_newHeaders;
    d_scriptBlob = d_newScriptBlob;
    numUTXOs = compactedHeaders.size();
    scriptBlobUsed = newScriptOffset;
    totalFreeSpace = 0;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    LogMsg("Compaction completed in " + std::to_string(duration.count()) + " ms");
    LogMsg("New UTXO count: " + std::to_string(numUTXOs));
    LogMsg("New script blob usage: " + std::to_string(scriptBlobUsed / (1024*1024)) + " MB");
    
    return true;
}

} // namespace gpu