#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_utils.h"
#include "gpu_logging.h"
#include <coins.h>
#include <primitives/transaction.h>
#include <script/script.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <memory>

namespace gpu {

// Helper function for logging
static inline void LogMsg(const std::string& msg) {
    ::LogGPUDebug(msg.c_str());
}

// Get or add txid to the table
static uint32_t GetOrAddTxid(const Txid& txid, 
                             std::unordered_map<Txid, uint32_t, SaltedTxidHasher>& txidMap,
                             std::vector<uint256_gpu>& uniqueTxids) {
    auto it = txidMap.find(txid);
    if (it != txidMap.end()) {
        return it->second;
    }
    
    uint32_t index = uniqueTxids.size();
    txidMap[txid] = index;
    uniqueTxids.push_back(ToGPU(txid));
    return index;
}

// CPU-side loader that interfaces with Bitcoin Core
bool LoadUTXOSetToCPU(const void* coinsCache, 
                      std::vector<UTXOHeader>& headers,
                      std::vector<uint256_gpu>& uniqueTxids,
                      std::vector<uint8_t>& scriptBlob,
                      size_t maxUTXOs,
                      size_t maxScriptBlobSize) {
    
    LogMsg("Starting UTXO set load from CPU");
    auto startTime = std::chrono::high_resolution_clock::now();
    
    headers.clear();
    uniqueTxids.clear();
    scriptBlob.clear();
    
    headers.reserve(maxUTXOs);
    uniqueTxids.reserve(50000000);  // Reserve for ~50M unique txids
    scriptBlob.reserve(maxScriptBlobSize);
    
    // Txid deduplication map
    std::unordered_map<Txid, uint32_t, SaltedTxidHasher> txidMap;
    
    uint32_t utxoCount = 0;
    uint64_t scriptBlobOffset = 0;
    uint64_t totalAmount = 0;
    
    // Statistics
    std::unordered_map<ScriptType, uint32_t> scriptTypeCount;
    
    // Get cursor to iterate through UTXO set
    std::unique_ptr<CCoinsViewCursor> cursor(const_cast<CCoinsViewCache*>(static_cast<const CCoinsViewCache*>(coinsCache))->Cursor());
    if (!cursor) {
        LogMsg("Failed to get coins cursor");
        return false;
    }
    
    LogMsg("Starting UTXO iteration...");
    
    // Process all UTXOs
    while (cursor->Valid() && utxoCount < maxUTXOs) {
        COutPoint outpoint;
        Coin coin;
        
        if (!cursor->GetKey(outpoint)) {
            LogMsg("Failed to get outpoint at index " + std::to_string(utxoCount));
            cursor->Next();
            continue;
        }
        
        if (!cursor->GetValue(coin)) {
            LogMsg("Failed to get coin at index " + std::to_string(utxoCount));
            cursor->Next();
            continue;
        }
        
        // Skip spent coins
        if (coin.IsSpent()) {
            cursor->Next();
            continue;
        }
        
        // Get or assign txid index
        uint32_t txidIndex = GetOrAddTxid(outpoint.hash, txidMap, uniqueTxids);
        
        // Prepare header
        UTXOHeader header;
        memset(&header, 0, sizeof(UTXOHeader));  // Clear padding
        
        header.amount = coin.out.nValue;
        header.blockHeight = coin.nHeight & 0xFFFFFF;  // 24 bits
        header.txid_index = txidIndex;
        header.vout = outpoint.n & 0xFFFF;  // 16 bits
        header.flags = coin.IsCoinBase() ? UTXO_FLAG_COINBASE : 0;
        header.script_size = coin.out.scriptPubKey.size();
        
        // Identify script type
        header.script_type = IdentifyScriptType(
            coin.out.scriptPubKey.data(),
            coin.out.scriptPubKey.size()
        );
        
        scriptTypeCount[static_cast<ScriptType>(header.script_type)]++;
        
        // Check if script fits in blob
        if (scriptBlobOffset + header.script_size > maxScriptBlobSize) {
            LogMsg("Script blob full at UTXO " + std::to_string(utxoCount));
            break;
        }
        
        header.script_offset = scriptBlobOffset;
        
        // Copy script data to blob
        size_t oldSize = scriptBlob.size();
        scriptBlob.resize(oldSize + header.script_size);
        memcpy(scriptBlob.data() + oldSize, 
               coin.out.scriptPubKey.data(),
               header.script_size);
        
        scriptBlobOffset += header.script_size;
        
        headers.push_back(header);
        totalAmount += coin.out.nValue;
        utxoCount++;
        
        // Log progress every 1M UTXOs
        if (utxoCount % 1000000 == 0) {
            LogMsg("Loaded " + std::to_string(utxoCount) + " UTXOs, " +
                   std::to_string(uniqueTxids.size()) + " unique txids, " +
                   std::to_string(scriptBlobOffset / (1024*1024)) + " MB scripts");
        }
        
        cursor->Next();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    LogMsg("UTXO loading completed in " + std::to_string(duration.count()) + " ms");
    LogMsg("Loaded " + std::to_string(headers.size()) + " UTXOs");
    LogMsg("Unique txids: " + std::to_string(uniqueTxids.size()));
    LogMsg("Script blob used: " + std::to_string(scriptBlobOffset / (1024*1024)) + " MB");
    LogMsg("Total value: " + std::to_string(totalAmount / 100000000.0) + " BTC");
    
    // Log script type distribution
    LogMsg("Script type distribution:");
    for (const auto& [type, count] : scriptTypeCount) {
        std::string typeName;
        switch(type) {
            case SCRIPT_TYPE_P2PKH: typeName = "P2PKH"; break;
            case SCRIPT_TYPE_P2WPKH: typeName = "P2WPKH"; break;
            case SCRIPT_TYPE_P2SH: typeName = "P2SH"; break;
            case SCRIPT_TYPE_P2WSH: typeName = "P2WSH"; break;
            case SCRIPT_TYPE_P2TR: typeName = "P2TR"; break;
            default: typeName = "NONSTANDARD"; break;
        }
        LogMsg("  " + typeName + ": " + std::to_string(count));
    }
    
    return true;
}

} // namespace gpu