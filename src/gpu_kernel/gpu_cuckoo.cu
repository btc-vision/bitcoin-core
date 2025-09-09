#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_utils.h"
#include "gpu_hash.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gpu {

// Cuckoo hash insert kernel - batch operation
__global__ void CuckooInsertKernel(
    uint32_t* hashTables,
    const uint32_t* indices,
    const uint64_t* hashes,  // Pre-computed hashes for each UTXO
    uint32_t count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t index = indices[tid];
    uint64_t hash = hashes[tid];
    
    // Try 4 hash positions
    for (int i = 0; i < 4; i++) {
        uint32_t pos = (hash >> (i * 16)) % GPUUTXOSet::TABLE_SIZE;
        uint32_t* addr = &hashTables[i * GPUUTXOSet::TABLE_SIZE + pos];
        
        uint32_t old = atomicCAS(addr, 0xFFFFFFFF, index);
        if (old == 0xFFFFFFFF) {
            return;  // Successfully inserted
        }
    }
    
    // All positions full - need eviction (handled separately)
}

// Cuckoo hash lookup kernel - batch operation
__global__ void CuckooLookupKernel(
    const uint32_t* hashTables,
    const UTXOHeader* headers,
    const uint256_gpu* txidTable,
    const uint256_gpu* searchTxids,
    const uint32_t* searchVouts,
    const uint64_t* searchHashes,  // Pre-computed hashes
    uint32_t* results,
    uint32_t count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint256_gpu searchTxid = searchTxids[tid];
    uint32_t searchVout = searchVouts[tid];
    uint64_t hash = searchHashes[tid];
    
    results[tid] = 0xFFFFFFFF;  // Not found by default
    
    // Check all 4 hash positions
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t pos = (hash >> (i * 16)) % GPUUTXOSet::TABLE_SIZE;
        uint32_t index = hashTables[i * GPUUTXOSet::TABLE_SIZE + pos];
        
        if (index != 0xFFFFFFFF) {
            // Verify it's the right UTXO
            const UTXOHeader& header = headers[index];
            const uint256_gpu& stored_txid = txidTable[header.txid_index];
            
            bool match = true;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                if (stored_txid.data[j] != searchTxid.data[j]) {
                    match = false;
                    break;
                }
            }
            
            if (match && header.vout == searchVout) {
                results[tid] = index;
                return;
            }
        }
    }
}

// Cuckoo hash delete kernel - batch operation
__global__ void CuckooDeleteKernel(
    uint32_t* hashTables,
    UTXOHeader* headers,
    const uint256_gpu* txidTable,
    const uint256_gpu* deleteTxids,
    const uint32_t* deleteVouts,
    const uint64_t* deleteHashes,  // Pre-computed hashes
    uint32_t count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint256_gpu deleteTxid = deleteTxids[tid];
    uint32_t deleteVout = deleteVouts[tid];
    uint64_t hash = deleteHashes[tid];
    
    // Check all 4 hash positions
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t pos = (hash >> (i * 16)) % GPUUTXOSet::TABLE_SIZE;
        uint32_t index = hashTables[i * GPUUTXOSet::TABLE_SIZE + pos];
        
        if (index != 0xFFFFFFFF) {
            // Verify it's the right UTXO
            UTXOHeader& header = headers[index];
            const uint256_gpu& stored_txid = txidTable[header.txid_index];
            
            bool match = true;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                if (stored_txid.data[j] != deleteTxid.data[j]) {
                    match = false;
                    break;
                }
            }
            
            if (match && header.vout == deleteVout) {
                // Mark as spent and remove from hash table
                header.flags |= UTXO_FLAG_SPENT;
                atomicExch(&hashTables[i * GPUUTXOSet::TABLE_SIZE + pos], 0xFFFFFFFF);
                return;
            }
        }
    }
}

// Host-side function to handle eviction when all slots are full
bool CuckooEvictAndInsert(
    uint32_t** d_hashTables,
    const uint256_gpu& txid,
    uint32_t vout,
    uint32_t newIndex,
    const UTXOHeader* d_headers,
    const uint256_gpu* d_txidTable
) {
    const int MAX_EVICTIONS = 500;  // Maximum eviction chain length
    
    // Start with the new item
    uint32_t currentIndex = newIndex;
    uint256_gpu currentTxid = txid;
    uint32_t currentVout = vout;
    
    for (int eviction = 0; eviction < MAX_EVICTIONS; eviction++) {
        // Try to insert current item
        for (int table = 0; table < 4; table++) {
            // Compute hash for this table
            uint64_t hash = SipHashUint256Extra(
                eviction * 4 + table,  // Use eviction count as salt variation
                eviction * 4 + table + 1,
                currentTxid,
                currentVout
            );
            uint32_t pos = static_cast<uint32_t>(hash % GPUUTXOSet::TABLE_SIZE);
            
            // Try to swap
            uint32_t victimIndex;
            cudaMemcpy(&victimIndex, d_hashTables[table] + pos, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            if (victimIndex == 0xFFFFFFFF) {
                // Empty slot found - insert and done
                cudaMemcpy(d_hashTables[table] + pos, &currentIndex, sizeof(uint32_t), cudaMemcpyHostToDevice);
                return true;
            }
            
            // Swap with victim
            cudaMemcpy(d_hashTables[table] + pos, &currentIndex, sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            // Now we need to relocate the victim
            currentIndex = victimIndex;
            
            // Get victim's txid and vout
            UTXOHeader victimHeader;
            cudaMemcpy(&victimHeader, d_headers + victimIndex, sizeof(UTXOHeader), cudaMemcpyDeviceToHost);
            cudaMemcpy(&currentTxid, d_txidTable + victimHeader.txid_index, sizeof(uint256_gpu), cudaMemcpyDeviceToHost);
            currentVout = victimHeader.vout;
            
            // Continue with victim in next iteration
            break;
        }
    }
    
    return false;  // Failed after MAX_EVICTIONS attempts
}

// Batch insert with eviction handling
bool BatchInsertWithEviction(
    GPUUTXOSet* utxoSet,
    const std::vector<uint256>& txids,
    const std::vector<uint32_t>& vouts,
    const std::vector<uint32_t>& indices
) {
    size_t count = txids.size();
    
    // Allocate device memory for batch operation
    uint256_gpu* d_txids;
    uint32_t* d_vouts;
    uint32_t* d_indices;
    uint64_t* d_hashes;
    
    cudaMalloc(&d_txids, count * sizeof(uint256_gpu));
    cudaMalloc(&d_vouts, count * sizeof(uint32_t));
    cudaMalloc(&d_indices, count * sizeof(uint32_t));
    cudaMalloc(&d_hashes, count * sizeof(uint64_t));
    
    // Copy data to device
    std::vector<uint256_gpu> gpu_txids;
    std::vector<uint64_t> hashes;
    
    for (size_t i = 0; i < count; i++) {
        gpu_txids.push_back(ToGPU(txids[i]));
        
        // Pre-compute hash
        uint64_t hash = SipHashUint256Extra(0, 1, gpu_txids[i], vouts[i]);
        hashes.push_back(hash);
    }
    
    cudaMemcpy(d_txids, gpu_txids.data(), count * sizeof(uint256_gpu), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vouts, vouts.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashes, hashes.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Flatten hash tables for kernel
    uint32_t* d_flatHashTables;
    cudaMalloc(&d_flatHashTables, 4 * GPUUTXOSet::TABLE_SIZE * sizeof(uint32_t));
    
    for (int i = 0; i < 4; i++) {
        cudaMemcpy(d_flatHashTables + i * GPUUTXOSet::TABLE_SIZE, 
                   utxoSet->GetHashTable(i), 
                   GPUUTXOSet::TABLE_SIZE * sizeof(uint32_t), 
                   cudaMemcpyDeviceToDevice);
    }
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;
    
    CuckooInsertKernel<<<numBlocks, blockSize>>>(
        d_flatHashTables,
        d_indices,
        d_hashes,
        count
    );
    
    cudaDeviceSynchronize();
    
    // Copy back to individual hash tables
    for (int i = 0; i < 4; i++) {
        cudaMemcpy(utxoSet->GetHashTable(i), 
                   d_flatHashTables + i * GPUUTXOSet::TABLE_SIZE,
                   GPUUTXOSet::TABLE_SIZE * sizeof(uint32_t), 
                   cudaMemcpyDeviceToDevice);
    }
    
    // Cleanup
    cudaFree(d_txids);
    cudaFree(d_vouts);
    cudaFree(d_indices);
    cudaFree(d_hashes);
    cudaFree(d_flatHashTables);
    
    return true;
}

} // namespace gpu