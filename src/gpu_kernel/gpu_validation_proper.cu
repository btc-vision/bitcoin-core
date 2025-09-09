// GPU Script Validation with Proper Hash Functions
#include "gpu_utxo.h"
#include "gpu_types.h"
#include "gpu_hash.cuh"
#include <cuda_runtime.h>
#include <cstring>

namespace gpu {

// P2PKH Script Validation
// scriptSig: <sig> <pubkey>
// scriptPubKey: OP_DUP OP_HASH160 <pubkeyhash> OP_EQUALVERIFY OP_CHECKSIG
__global__ void ValidateP2PKHKernelProper(
    const UTXOHeader* headers,
    const uint8_t* scriptBlob,
    const uint32_t* utxoIndices,
    const uint8_t* signatures,     // Array of signatures
    const uint8_t* pubkeys,         // Array of public keys
    bool* results,
    uint32_t count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t utxoIndex = utxoIndices[tid];
    const UTXOHeader& header = headers[utxoIndex];
    
    // Check if it's actually P2PKH
    if (header.script_type != SCRIPT_TYPE_P2PKH) {
        results[tid] = false;
        return;
    }
    
    // Get the public key hash from scriptPubKey
    const uint8_t* script = scriptBlob + header.script_offset;
    if (header.script_size != 25) {
        results[tid] = false;
        return;
    }
    
    // Verify script structure: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
    if (script[0] != 0x76 || script[1] != 0xa9 || script[2] != 0x14 ||
        script[23] != 0x88 || script[24] != 0xac) {
        results[tid] = false;
        return;
    }
    
    // Extract the expected pubkey hash
    uint8_t expected_hash[20];
    memcpy(expected_hash, script + 3, 20);
    
    // Get the actual public key for this transaction
    const uint8_t* pubkey = pubkeys + tid * 33;  // Assuming compressed pubkeys
    
    // Compute Hash160 of the public key
    uint8_t computed_hash[20];
    hash160(pubkey, 33, computed_hash);
    
    // Compare hashes
    bool hash_match = true;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        if (computed_hash[i] != expected_hash[i]) {
            hash_match = false;
            break;
        }
    }
    
    results[tid] = hash_match;
    
    // Note: Actual signature verification would require ECDSA implementation
    // For now, we're just validating the script structure and pubkey hash
}

// P2WPKH (Segregated Witness Pay-to-Witness-Public-Key-Hash)
__global__ void ValidateP2WPKHKernelProper(
    const UTXOHeader* headers,
    const uint8_t* scriptBlob,
    const uint32_t* utxoIndices,
    const uint8_t* witnesses,      // Witness data
    bool* results,
    uint32_t count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t utxoIndex = utxoIndices[tid];
    const UTXOHeader& header = headers[utxoIndex];
    
    // Check if it's actually P2WPKH
    if (header.script_type != SCRIPT_TYPE_P2WPKH) {
        results[tid] = false;
        return;
    }
    
    // P2WPKH scriptPubKey: OP_0 <20-byte-key-hash>
    const uint8_t* script = scriptBlob + header.script_offset;
    if (header.script_size != 22) {
        results[tid] = false;
        return;
    }
    
    if (script[0] != 0x00 || script[1] != 0x14) {
        results[tid] = false;
        return;
    }
    
    // Extract witness data (signature and pubkey)
    const uint8_t* witness = witnesses + tid * 107;  // 72 byte sig + 33 byte pubkey + 2 byte lengths
    
    // For P2WPKH, witness stack should have exactly 2 items: signature and pubkey
    uint8_t sig_len = witness[0];
    const uint8_t* signature = witness + 1;
    uint8_t pubkey_len = witness[1 + sig_len];
    const uint8_t* pubkey = witness + 2 + sig_len;
    
    // Compute Hash160 of the public key
    uint8_t computed_hash[20];
    hash160(pubkey, pubkey_len, computed_hash);
    
    // Compare with the hash in the script
    bool hash_match = true;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        if (computed_hash[i] != script[2 + i]) {
            hash_match = false;
            break;
        }
    }
    
    results[tid] = hash_match;
}

// P2SH (Pay-to-Script-Hash) validation
__global__ void ValidateP2SHKernelProper(
    const UTXOHeader* headers,
    const uint8_t* scriptBlob,
    const uint32_t* utxoIndices,
    const uint8_t* redeemScripts,   // The actual scripts being redeemed
    bool* results,
    uint32_t count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t utxoIndex = utxoIndices[tid];
    const UTXOHeader& header = headers[utxoIndex];
    
    // Check if it's actually P2SH
    if (header.script_type != SCRIPT_TYPE_P2SH) {
        results[tid] = false;
        return;
    }
    
    // P2SH scriptPubKey: OP_HASH160 <20-byte-script-hash> OP_EQUAL
    const uint8_t* script = scriptBlob + header.script_offset;
    if (header.script_size != 23) {
        results[tid] = false;
        return;
    }
    
    if (script[0] != 0xa9 || script[1] != 0x14 || script[22] != 0x87) {
        results[tid] = false;
        return;
    }
    
    // Get the redeem script for this transaction
    const uint8_t* redeem_script = redeemScripts + tid * 520;  // Max standard redeem script size
    uint16_t redeem_script_len = *((uint16_t*)redeem_script);
    const uint8_t* redeem_script_data = redeem_script + 2;
    
    // Compute Hash160 of the redeem script
    uint8_t computed_hash[20];
    hash160(redeem_script_data, redeem_script_len, computed_hash);
    
    // Compare with the hash in the P2SH script
    bool hash_match = true;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        if (computed_hash[i] != script[2 + i]) {
            hash_match = false;
            break;
        }
    }
    
    results[tid] = hash_match;
    
    // Note: Would also need to execute the redeem script itself
}

// P2TR (Taproot) validation
__global__ void ValidateTaprootKernelProper(
    const UTXOHeader* headers,
    const uint8_t* scriptBlob,
    const uint32_t* utxoIndices,
    const uint8_t* witnesses,
    bool* results,
    uint32_t count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t utxoIndex = utxoIndices[tid];
    const UTXOHeader& header = headers[utxoIndex];
    
    // Check if it's actually P2TR
    if (header.script_type != SCRIPT_TYPE_P2TR) {
        results[tid] = false;
        return;
    }
    
    // P2TR scriptPubKey: OP_1 <32-byte-taproot-output>
    const uint8_t* script = scriptBlob + header.script_offset;
    if (header.script_size != 34) {
        results[tid] = false;
        return;
    }
    
    if (script[0] != 0x51 || script[1] != 0x20) {
        results[tid] = false;
        return;
    }
    
    // Extract the taproot output point
    uint8_t taproot_output[32];
    memcpy(taproot_output, script + 2, 32);
    
    // Taproot validation is complex and involves Schnorr signatures
    // For now, just validate the script structure
    results[tid] = true;
}

// Batch validation kernel that dispatches to appropriate validation
__global__ void ValidateBatchKernelProper(
    const UTXOHeader* headers,
    const uint8_t* scriptBlob,
    const uint32_t* utxoIndices,
    const uint8_t* validationData,  // Generic validation data
    bool* results,
    uint32_t count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t utxoIndex = utxoIndices[tid];
    const UTXOHeader& header = headers[utxoIndex];
    
    // Dispatch based on script type
    switch(header.script_type) {
        case SCRIPT_TYPE_P2PKH:
            // Validate P2PKH
            {
                const uint8_t* script = scriptBlob + header.script_offset;
                if (header.script_size == 25 &&
                    script[0] == 0x76 && script[1] == 0xa9 && script[2] == 0x14 &&
                    script[23] == 0x88 && script[24] == 0xac) {
                    results[tid] = true;
                } else {
                    results[tid] = false;
                }
            }
            break;
            
        case SCRIPT_TYPE_P2WPKH:
            // Validate P2WPKH
            {
                const uint8_t* script = scriptBlob + header.script_offset;
                results[tid] = (header.script_size == 22 && 
                               script[0] == 0x00 && script[1] == 0x14);
            }
            break;
            
        case SCRIPT_TYPE_P2SH:
            // Validate P2SH
            {
                const uint8_t* script = scriptBlob + header.script_offset;
                results[tid] = (header.script_size == 23 &&
                               script[0] == 0xa9 && script[1] == 0x14 && 
                               script[22] == 0x87);
            }
            break;
            
        case SCRIPT_TYPE_P2WSH:
            // Validate P2WSH
            {
                const uint8_t* script = scriptBlob + header.script_offset;
                results[tid] = (header.script_size == 34 &&
                               script[0] == 0x00 && script[1] == 0x20);
            }
            break;
            
        case SCRIPT_TYPE_P2TR:
            // Validate Taproot
            {
                const uint8_t* script = scriptBlob + header.script_offset;
                results[tid] = (header.script_size == 34 &&
                               script[0] == 0x51 && script[1] == 0x20);
            }
            break;
            
        default:
            results[tid] = false;
            break;
    }
}

// Host-side function to launch validation
extern "C" bool LaunchValidationProper(
    const GPUUTXOSet& utxoSet,
    const uint32_t* utxoIndices,
    const uint8_t* validationData,
    bool* results,
    uint32_t count,
    cudaStream_t stream
) {
    // Determine grid and block dimensions
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    
    // Launch the batch validation kernel
    ValidateBatchKernelProper<<<gridSize, blockSize, 0, stream>>>(
        nullptr,  // Will be set from utxoSet
        nullptr,  // Will be set from utxoSet
        utxoIndices,
        validationData,
        results,
        count
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return false;
    }
    
    // Synchronize if needed
    if (stream == 0) {
        cudaDeviceSynchronize();
    }
    
    return true;
}

} // namespace gpu