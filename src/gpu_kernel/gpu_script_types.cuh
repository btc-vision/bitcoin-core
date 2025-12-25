// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SCRIPT_TYPES_CUH
#define BITCOIN_GPU_KERNEL_GPU_SCRIPT_TYPES_CUH

#include "gpu_types.h"
#include <cstdint>
#include <cstring>

namespace gpu {

// ============================================================================
// Script Execution Limits (must match Bitcoin Core consensus rules)
// ============================================================================

// Stack limits - 520 bytes per element for ALL sigversions
// The 520-byte limit applies to stack elements (sigs, pubkeys, data)
// The tapscript itself is passed directly to EvalScript and can be larger
constexpr uint32_t MAX_STACK_SIZE = 1000;
constexpr uint32_t MAX_STACK_ELEMENT_SIZE = 520;            // 520 bytes for all sigversions
constexpr uint32_t MAX_SCRIPT_SIZE = 10000;
constexpr uint32_t MAX_SCRIPT_ELEMENT_SIZE = 520;           // Legacy push limit check
constexpr uint32_t MAX_OPS_PER_SCRIPT = 201;
constexpr uint32_t MAX_PUBKEYS_PER_MULTISIG = 20;
constexpr int64_t MAX_SCRIPT_NUM_LENGTH = 4;  // Default, 5 for locktime

// GPU-optimized stack sizes for different script complexity levels
// Simple scripts (P2WPKH, P2TR key-path, P2PKH) use small local stacks
// Complex scripts (multisig, P2WSH) fall back to global memory
constexpr uint32_t GPU_SMALL_STACK_SIZE = 16;   // For simple scripts (~17KB per context)

// Tapscript limits (BIP342)
// - The 520-byte element size limit is REMOVED
// - The 201 opcode limit is REMOVED (replaced by sigop budget)
// - The 10,000 byte script size limit is REMOVED
constexpr int64_t TAPSCRIPT_VALIDATION_WEIGHT_PER_SIGOP = 50;
constexpr int64_t TAPSCRIPT_MAX_VALIDATION_WEIGHT = 4000000; // 4M weight units

// ============================================================================
// Script Error Codes (must match ScriptError_t in script_error.h exactly)
// ============================================================================

enum GPUScriptError : uint8_t {
    GPU_SCRIPT_ERR_OK = 0,
    GPU_SCRIPT_ERR_UNKNOWN_ERROR,
    GPU_SCRIPT_ERR_EVAL_FALSE,
    GPU_SCRIPT_ERR_OP_RETURN,

    // Max sizes
    GPU_SCRIPT_ERR_SCRIPT_SIZE,
    GPU_SCRIPT_ERR_PUSH_SIZE,
    GPU_SCRIPT_ERR_OP_COUNT,
    GPU_SCRIPT_ERR_STACK_SIZE,
    GPU_SCRIPT_ERR_SIG_COUNT,
    GPU_SCRIPT_ERR_PUBKEY_COUNT,

    // Failed verify operations
    GPU_SCRIPT_ERR_VERIFY,
    GPU_SCRIPT_ERR_EQUALVERIFY,
    GPU_SCRIPT_ERR_CHECKMULTISIGVERIFY,
    GPU_SCRIPT_ERR_CHECKSIGVERIFY,
    GPU_SCRIPT_ERR_NUMEQUALVERIFY,

    // Logical/Format/Canonical errors
    GPU_SCRIPT_ERR_BAD_OPCODE,
    GPU_SCRIPT_ERR_DISABLED_OPCODE,
    GPU_SCRIPT_ERR_INVALID_STACK_OPERATION,
    GPU_SCRIPT_ERR_INVALID_ALTSTACK_OPERATION,
    GPU_SCRIPT_ERR_UNBALANCED_CONDITIONAL,

    // CHECKLOCKTIMEVERIFY and CHECKSEQUENCEVERIFY
    GPU_SCRIPT_ERR_NEGATIVE_LOCKTIME,
    GPU_SCRIPT_ERR_UNSATISFIED_LOCKTIME,

    // Malleability
    GPU_SCRIPT_ERR_SIG_HASHTYPE,
    GPU_SCRIPT_ERR_SIG_DER,
    GPU_SCRIPT_ERR_SIG_ECDSA,       // ECDSA signature verification failed
    GPU_SCRIPT_ERR_MINIMALDATA,
    GPU_SCRIPT_ERR_SIG_PUSHONLY,
    GPU_SCRIPT_ERR_SIG_HIGH_S,
    GPU_SCRIPT_ERR_SIG_NULLDUMMY,
    GPU_SCRIPT_ERR_PUBKEYTYPE,
    GPU_SCRIPT_ERR_CLEANSTACK,
    GPU_SCRIPT_ERR_MINIMALIF,
    GPU_SCRIPT_ERR_SIG_NULLFAIL,

    // Softfork safeness
    GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_NOPS,
    GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM,
    GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION,
    GPU_SCRIPT_ERR_DISCOURAGE_OP_SUCCESS,
    GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_PUBKEYTYPE,

    // Segregated witness
    GPU_SCRIPT_ERR_WITNESS_PROGRAM_WRONG_LENGTH,
    GPU_SCRIPT_ERR_WITNESS_PROGRAM_WITNESS_EMPTY,
    GPU_SCRIPT_ERR_WITNESS_PROGRAM_MISMATCH,
    GPU_SCRIPT_ERR_WITNESS_MALLEATED,
    GPU_SCRIPT_ERR_WITNESS_MALLEATED_P2SH,
    GPU_SCRIPT_ERR_WITNESS_UNEXPECTED,
    GPU_SCRIPT_ERR_WITNESS_PUBKEYTYPE,

    // Taproot
    GPU_SCRIPT_ERR_SCHNORR_SIG_SIZE,
    GPU_SCRIPT_ERR_SCHNORR_SIG_HASHTYPE,
    GPU_SCRIPT_ERR_SCHNORR_SIG,
    GPU_SCRIPT_ERR_TAPROOT_WRONG_CONTROL_SIZE,
    GPU_SCRIPT_ERR_TAPSCRIPT_VALIDATION_WEIGHT,
    GPU_SCRIPT_ERR_TAPSCRIPT_CHECKMULTISIG,
    GPU_SCRIPT_ERR_TAPSCRIPT_MINIMALIF,
    GPU_SCRIPT_ERR_TAPSCRIPT_EMPTY_PUBKEY,

    // Constant scriptCode
    GPU_SCRIPT_ERR_OP_CODESEPARATOR,
    GPU_SCRIPT_ERR_SIG_FINDANDDELETE,

    GPU_SCRIPT_ERR_ERROR_COUNT
};

// ============================================================================
// Signature Version (must match SigVersion in interpreter.h)
// ============================================================================

enum GPUSigVersion : uint8_t {
    GPU_SIGVERSION_BASE = 0,       // Bare scripts and BIP16 P2SH-wrapped redeemscripts
    GPU_SIGVERSION_WITNESS_V0 = 1, // Witness v0 (P2WPKH and P2WSH); see BIP 141
    GPU_SIGVERSION_TAPROOT = 2,    // Witness v1 key path spending; see BIP 341
    GPU_SIGVERSION_TAPSCRIPT = 3,  // Witness v1 script path spending; see BIP 342
};

// ============================================================================
// Script Verification Flags (bit positions matching Bitcoin Core)
// ============================================================================

constexpr uint32_t GPU_SCRIPT_VERIFY_NONE                          = 0;
constexpr uint32_t GPU_SCRIPT_VERIFY_P2SH                          = (1U << 0);
constexpr uint32_t GPU_SCRIPT_VERIFY_STRICTENC                     = (1U << 1);
constexpr uint32_t GPU_SCRIPT_VERIFY_DERSIG                        = (1U << 2);
constexpr uint32_t GPU_SCRIPT_VERIFY_LOW_S                         = (1U << 3);
constexpr uint32_t GPU_SCRIPT_VERIFY_NULLDUMMY                     = (1U << 4);
constexpr uint32_t GPU_SCRIPT_VERIFY_SIGPUSHONLY                   = (1U << 5);
constexpr uint32_t GPU_SCRIPT_VERIFY_MINIMALDATA                   = (1U << 6);
constexpr uint32_t GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS    = (1U << 7);
constexpr uint32_t GPU_SCRIPT_VERIFY_CLEANSTACK                    = (1U << 8);
constexpr uint32_t GPU_SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY           = (1U << 9);
constexpr uint32_t GPU_SCRIPT_VERIFY_CHECKSEQUENCEVERIFY           = (1U << 10);
constexpr uint32_t GPU_SCRIPT_VERIFY_WITNESS                       = (1U << 11);
constexpr uint32_t GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_WITNESS = (1U << 12);
constexpr uint32_t GPU_SCRIPT_VERIFY_MINIMALIF                     = (1U << 13);
constexpr uint32_t GPU_SCRIPT_VERIFY_NULLFAIL                      = (1U << 14);
constexpr uint32_t GPU_SCRIPT_VERIFY_WITNESS_PUBKEYTYPE            = (1U << 15);
constexpr uint32_t GPU_SCRIPT_VERIFY_CONST_SCRIPTCODE              = (1U << 16);
constexpr uint32_t GPU_SCRIPT_VERIFY_TAPROOT                       = (1U << 17);
constexpr uint32_t GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_TAPROOT = (1U << 18);
constexpr uint32_t GPU_SCRIPT_VERIFY_DISCOURAGE_OP_SUCCESS         = (1U << 19);
constexpr uint32_t GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_PUBKEY  = (1U << 20);

// ============================================================================
// SIGHASH Types
// ============================================================================

constexpr uint8_t GPU_SIGHASH_DEFAULT       = 0x00;  // Taproot only
constexpr uint8_t GPU_SIGHASH_ALL           = 0x01;
constexpr uint8_t GPU_SIGHASH_NONE          = 0x02;
constexpr uint8_t GPU_SIGHASH_SINGLE        = 0x03;
constexpr uint8_t GPU_SIGHASH_ANYONECANPAY  = 0x80;

// ============================================================================
// Stack Element (fixed-size for GPU memory efficiency)
// ============================================================================

struct GPUStackElement {
    uint8_t data[MAX_STACK_ELEMENT_SIZE];  // 10000 bytes for Tapscript support
    uint16_t size;                          // Actual size
    uint16_t padding;                       // Alignment to 4 bytes

    __host__ __device__ GPUStackElement() : size(0), padding(0) {
        // Don't zero data for performance - size=0 means empty
    }

    __host__ __device__ void clear() {
        size = 0;
    }

    __host__ __device__ bool empty() const {
        return size == 0;
    }

    __host__ __device__ void set(const uint8_t* src, uint16_t len) {
        if (len > MAX_STACK_ELEMENT_SIZE) len = MAX_STACK_ELEMENT_SIZE;
        memcpy(data, src, len);
        size = len;
    }
};

static_assert(sizeof(GPUStackElement) == 524, "GPUStackElement must be 524 bytes");

// ============================================================================
// Condition Stack (for IF/ELSE/ENDIF tracking)
// Optimized like CPU's ConditionStack - only tracks size and first false position
// ============================================================================

struct GPUConditionStack {
    uint32_t size;                          // Number of conditions on stack
    uint32_t first_false_pos;               // Position of first FALSE, or UINT32_MAX if all TRUE

    static constexpr uint32_t NO_FALSE = UINT32_MAX;

    __host__ __device__ GPUConditionStack() : size(0), first_false_pos(NO_FALSE) {}

    __host__ __device__ bool empty() const {
        return size == 0;
    }

    __host__ __device__ bool all_true() const {
        return first_false_pos == NO_FALSE;
    }

    __host__ __device__ void push_back(bool value) {
        if (first_false_pos == NO_FALSE && !value) {
            first_false_pos = size;
        }
        ++size;
    }

    __host__ __device__ void pop_back() {
        --size;
        if (first_false_pos >= size) {
            first_false_pos = NO_FALSE;
        }
    }

    __host__ __device__ void toggle_top() {
        if (first_false_pos == NO_FALSE) {
            // All true, toggling top makes it false
            first_false_pos = size - 1;
        } else if (first_false_pos == size - 1) {
            // Top is the first false, toggling makes everything true
            first_false_pos = NO_FALSE;
        }
        // else: There is a false value, but not on top. No action is needed
        // as toggling anything but the first false value is unobservable.
        // This matches Bitcoin Core's ConditionStack implementation exactly.
    }
};

// ============================================================================
// Precomputed Transaction Data (for sighash computation)
// Matches PrecomputedTransactionData in interpreter.h
// ============================================================================

struct GPUPrecomputedTxData {
    // BIP341 (Taproot) - single SHA256
    uint256_gpu prevouts_single_hash;
    uint256_gpu sequences_single_hash;
    uint256_gpu outputs_single_hash;
    uint256_gpu spent_amounts_single_hash;
    uint256_gpu spent_scripts_single_hash;
    bool bip341_ready;

    // BIP143 (SegWit v0) - double SHA256
    uint256_gpu hashPrevouts;
    uint256_gpu hashSequence;
    uint256_gpu hashOutputs;
    bool bip143_ready;

    __host__ __device__ GPUPrecomputedTxData()
        : bip341_ready(false), bip143_ready(false) {}
};

// ============================================================================
// Script Execution Data (for Tapscript)
// Matches ScriptExecutionData in interpreter.h
// ============================================================================

struct GPUScriptExecutionData {
    // Tapleaf hash
    uint256_gpu tapleaf_hash;
    bool tapleaf_hash_init;

    // Code separator position
    uint32_t codeseparator_pos;
    bool codeseparator_pos_init;

    // Annex
    uint256_gpu annex_hash;
    bool annex_present;
    bool annex_init;

    // Validation weight tracking (Tapscript)
    int64_t validation_weight_left;
    bool validation_weight_init;

    __host__ __device__ GPUScriptExecutionData()
        : tapleaf_hash_init(false)
        , codeseparator_pos(0xFFFFFFFF)
        , codeseparator_pos_init(false)
        , annex_present(false)
        , annex_init(false)
        , validation_weight_left(0)
        , validation_weight_init(false) {}
};

// ============================================================================
// Script Execution Context (per-thread state for GPU script execution)
// This is the main structure each GPU thread uses during EvalScript
// ============================================================================

struct GPUScriptContext {
    // ========== Main Stack ==========
    GPUStackElement stack[MAX_STACK_SIZE];
    uint16_t stack_size;

    // ========== Alternate Stack ==========
    GPUStackElement altstack[MAX_STACK_SIZE];
    uint16_t altstack_size;

    // ========== Condition Stack ==========
    GPUConditionStack conditions;

    // ========== Execution State ==========
    uint32_t pc;                  // Program counter (byte offset in script)
    uint32_t opcode_count;        // Number of opcodes executed
    uint32_t codeseparator_pos;   // Position of last OP_CODESEPARATOR

    // ========== Configuration ==========
    uint32_t verify_flags;        // SCRIPT_VERIFY_* flags
    GPUSigVersion sigversion;     // Signature version (BASE, WITNESS_V0, etc.)

    // ========== Error State ==========
    GPUScriptError error;         // Error code if execution failed
    bool success;                 // True if execution succeeded

    // ========== Precomputed Data ==========
    GPUPrecomputedTxData txdata;
    GPUScriptExecutionData execdata;

    // ========== Transaction Context ==========
    int32_t tx_version;
    uint32_t tx_locktime;
    uint32_t input_index;         // Index of the input being verified
    uint32_t input_sequence;      // Sequence of the input being verified
    int64_t input_amount;         // Amount of the input being verified

    // ========== Precomputed Sighash (for batch validation) ==========
    uint256_gpu precomputed_sighash;
    bool precomputed_sighash_valid;

    __host__ __device__ GPUScriptContext()
        : stack_size(0)
        , altstack_size(0)
        , pc(0)
        , opcode_count(0)
        , codeseparator_pos(0)
        , verify_flags(0)
        , sigversion(GPU_SIGVERSION_BASE)
        , error(GPU_SCRIPT_ERR_OK)
        , success(false)
        , tx_version(0)
        , tx_locktime(0)
        , input_index(0)
        , input_sequence(0)
        , input_amount(0)
        , precomputed_sighash_valid(false) {}

    __host__ __device__ void reset() {
        stack_size = 0;
        altstack_size = 0;
        conditions = GPUConditionStack();
        pc = 0;
        opcode_count = 0;
        codeseparator_pos = 0;
        error = GPU_SCRIPT_ERR_OK;
        success = false;
    }

    __host__ __device__ bool set_error(GPUScriptError err) {
        error = err;
        success = false;
        return false;
    }

    __host__ __device__ bool set_success() {
        error = GPU_SCRIPT_ERR_OK;
        success = true;
        return true;
    }
};

// ============================================================================
// Small Script Context (for simple scripts - uses local memory)
// P2WPKH, P2TR key-path, P2PKH only need 2-5 stack elements
// ============================================================================

struct GPUScriptContextSmall {
    // Small stacks for simple scripts
    GPUStackElement stack[GPU_SMALL_STACK_SIZE];
    uint16_t stack_size;

    GPUStackElement altstack[GPU_SMALL_STACK_SIZE];
    uint16_t altstack_size;

    // Same execution state as full context
    GPUConditionStack conditions;
    uint32_t pc;
    uint32_t opcode_count;
    uint32_t codeseparator_pos;
    uint32_t verify_flags;
    GPUSigVersion sigversion;
    GPUScriptError error;
    bool success;

    GPUPrecomputedTxData txdata;
    GPUScriptExecutionData execdata;

    int32_t tx_version;
    uint32_t tx_locktime;
    int64_t input_amount;
    uint32_t input_sequence;
    uint32_t n_in;

    __host__ __device__ GPUScriptContextSmall() {
        stack_size = 0;
        altstack_size = 0;
        pc = 0;
        opcode_count = 0;
        codeseparator_pos = 0xFFFFFFFF;
        verify_flags = 0;
        sigversion = GPU_SIGVERSION_BASE;
        error = GPU_SCRIPT_ERR_OK;
        success = false;
    }

    __host__ __device__ bool set_error(GPUScriptError err) {
        error = err;
        success = false;
        return false;
    }

    __host__ __device__ bool set_success() {
        error = GPU_SCRIPT_ERR_OK;
        success = true;
        return true;
    }

    // Check if we're about to overflow small stack
    __host__ __device__ bool would_overflow(uint16_t additional = 1) const {
        return (stack_size + additional) > GPU_SMALL_STACK_SIZE;
    }

    __host__ __device__ bool altstack_would_overflow(uint16_t additional = 1) const {
        return (altstack_size + additional) > GPU_SMALL_STACK_SIZE;
    }
};

// ============================================================================
// Signature Verification Job (for batch signature verification)
// ============================================================================

struct GPUSignatureJob {
    // Input identification
    uint32_t tx_index;
    uint32_t input_index;

    // Signature type
    enum Type : uint8_t {
        SIG_ECDSA = 0,
        SIG_SCHNORR = 1
    } sig_type;

    // Sighash type
    uint8_t sighash_type;

    // Signature data
    uint8_t signature[73];        // Max DER signature size (ECDSA) or 65 (Schnorr)
    uint8_t sig_len;

    // Public key data
    uint8_t pubkey[65];           // Max uncompressed pubkey size
    uint8_t pubkey_len;

    // Precomputed sighash (32 bytes)
    uint256_gpu sighash;

    // Result
    bool verified;
    bool processed;

    __host__ __device__ GPUSignatureJob()
        : tx_index(0)
        , input_index(0)
        , sig_type(SIG_ECDSA)
        , sighash_type(GPU_SIGHASH_ALL)
        , sig_len(0)
        , pubkey_len(0)
        , verified(false)
        , processed(false) {}
};

// ============================================================================
// Transaction Input for GPU Validation
// ============================================================================

struct GPUTxInput {
    // Outpoint being spent
    uint256_gpu prev_txid;
    uint32_t prev_vout;

    // UTXO info
    uint32_t utxo_index;          // Index in GPU UTXO set
    int64_t amount;               // Satoshi value

    // ScriptSig
    uint32_t scriptsig_offset;    // Offset in script blob
    uint16_t scriptsig_size;

    // ScriptPubKey (from UTXO)
    uint32_t scriptpubkey_offset;
    uint16_t scriptpubkey_size;
    uint8_t script_type;          // ScriptType from gpu_utxo.h

    // Witness data
    uint32_t witness_offset;      // Offset in witness blob
    uint16_t witness_count;       // Number of witness stack elements
    uint32_t witness_total_size;  // Total witness data size (needs 32 bits for large tapscripts)

    // Transaction context
    uint32_t sequence;

    // Result
    GPUScriptError error;
    bool valid;
};

// ============================================================================
// Batch Validation Result
// ============================================================================

struct GPUValidationResult {
    uint32_t total_inputs;
    uint32_t valid_count;
    uint32_t invalid_count;
    uint32_t gpu_validated;       // Number validated on GPU (vs CPU fallback)

    // First error encountered (if any)
    uint32_t first_error_tx_idx;
    uint32_t first_error_input_idx;
    GPUScriptError first_error;
};

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SCRIPT_TYPES_CUH
