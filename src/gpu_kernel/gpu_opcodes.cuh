// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_OPCODES_CUH
#define BITCOIN_GPU_KERNEL_GPU_OPCODES_CUH

#include <cstdint>

namespace gpu {

// ============================================================================
// Script Opcodes (must match opcodetype in script.h exactly)
// ============================================================================

enum GPUOpcode : uint8_t {
    // Push value
    GPU_OP_0 = 0x00,
    GPU_OP_FALSE = GPU_OP_0,
    GPU_OP_PUSHDATA1 = 0x4c,
    GPU_OP_PUSHDATA2 = 0x4d,
    GPU_OP_PUSHDATA4 = 0x4e,
    GPU_OP_1NEGATE = 0x4f,
    GPU_OP_RESERVED = 0x50,
    GPU_OP_1 = 0x51,
    GPU_OP_TRUE = GPU_OP_1,
    GPU_OP_2 = 0x52,
    GPU_OP_3 = 0x53,
    GPU_OP_4 = 0x54,
    GPU_OP_5 = 0x55,
    GPU_OP_6 = 0x56,
    GPU_OP_7 = 0x57,
    GPU_OP_8 = 0x58,
    GPU_OP_9 = 0x59,
    GPU_OP_10 = 0x5a,
    GPU_OP_11 = 0x5b,
    GPU_OP_12 = 0x5c,
    GPU_OP_13 = 0x5d,
    GPU_OP_14 = 0x5e,
    GPU_OP_15 = 0x5f,
    GPU_OP_16 = 0x60,

    // Control
    GPU_OP_NOP = 0x61,
    GPU_OP_VER = 0x62,
    GPU_OP_IF = 0x63,
    GPU_OP_NOTIF = 0x64,
    GPU_OP_VERIF = 0x65,
    GPU_OP_VERNOTIF = 0x66,
    GPU_OP_ELSE = 0x67,
    GPU_OP_ENDIF = 0x68,
    GPU_OP_VERIFY = 0x69,
    GPU_OP_RETURN = 0x6a,

    // Stack ops
    GPU_OP_TOALTSTACK = 0x6b,
    GPU_OP_FROMALTSTACK = 0x6c,
    GPU_OP_2DROP = 0x6d,
    GPU_OP_2DUP = 0x6e,
    GPU_OP_3DUP = 0x6f,
    GPU_OP_2OVER = 0x70,
    GPU_OP_2ROT = 0x71,
    GPU_OP_2SWAP = 0x72,
    GPU_OP_IFDUP = 0x73,
    GPU_OP_DEPTH = 0x74,
    GPU_OP_DROP = 0x75,
    GPU_OP_DUP = 0x76,
    GPU_OP_NIP = 0x77,
    GPU_OP_OVER = 0x78,
    GPU_OP_PICK = 0x79,
    GPU_OP_ROLL = 0x7a,
    GPU_OP_ROT = 0x7b,
    GPU_OP_SWAP = 0x7c,
    GPU_OP_TUCK = 0x7d,

    // Splice ops (mostly disabled)
    GPU_OP_CAT = 0x7e,
    GPU_OP_SUBSTR = 0x7f,
    GPU_OP_LEFT = 0x80,
    GPU_OP_RIGHT = 0x81,
    GPU_OP_SIZE = 0x82,

    // Bit logic (mostly disabled)
    GPU_OP_INVERT = 0x83,
    GPU_OP_AND = 0x84,
    GPU_OP_OR = 0x85,
    GPU_OP_XOR = 0x86,
    GPU_OP_EQUAL = 0x87,
    GPU_OP_EQUALVERIFY = 0x88,
    GPU_OP_RESERVED1 = 0x89,
    GPU_OP_RESERVED2 = 0x8a,

    // Numeric
    GPU_OP_1ADD = 0x8b,
    GPU_OP_1SUB = 0x8c,
    GPU_OP_2MUL = 0x8d,
    GPU_OP_2DIV = 0x8e,
    GPU_OP_NEGATE = 0x8f,
    GPU_OP_ABS = 0x90,
    GPU_OP_NOT = 0x91,
    GPU_OP_0NOTEQUAL = 0x92,

    GPU_OP_ADD = 0x93,
    GPU_OP_SUB = 0x94,
    GPU_OP_MUL = 0x95,
    GPU_OP_DIV = 0x96,
    GPU_OP_MOD = 0x97,
    GPU_OP_LSHIFT = 0x98,
    GPU_OP_RSHIFT = 0x99,

    GPU_OP_BOOLAND = 0x9a,
    GPU_OP_BOOLOR = 0x9b,
    GPU_OP_NUMEQUAL = 0x9c,
    GPU_OP_NUMEQUALVERIFY = 0x9d,
    GPU_OP_NUMNOTEQUAL = 0x9e,
    GPU_OP_LESSTHAN = 0x9f,
    GPU_OP_GREATERTHAN = 0xa0,
    GPU_OP_LESSTHANOREQUAL = 0xa1,
    GPU_OP_GREATERTHANOREQUAL = 0xa2,
    GPU_OP_MIN = 0xa3,
    GPU_OP_MAX = 0xa4,

    GPU_OP_WITHIN = 0xa5,

    // Crypto
    GPU_OP_RIPEMD160 = 0xa6,
    GPU_OP_SHA1 = 0xa7,
    GPU_OP_SHA256 = 0xa8,
    GPU_OP_HASH160 = 0xa9,
    GPU_OP_HASH256 = 0xaa,
    GPU_OP_CODESEPARATOR = 0xab,
    GPU_OP_CHECKSIG = 0xac,
    GPU_OP_CHECKSIGVERIFY = 0xad,
    GPU_OP_CHECKMULTISIG = 0xae,
    GPU_OP_CHECKMULTISIGVERIFY = 0xaf,

    // Expansion
    GPU_OP_NOP1 = 0xb0,
    GPU_OP_CHECKLOCKTIMEVERIFY = 0xb1,
    GPU_OP_NOP2 = GPU_OP_CHECKLOCKTIMEVERIFY,
    GPU_OP_CHECKSEQUENCEVERIFY = 0xb2,
    GPU_OP_NOP3 = GPU_OP_CHECKSEQUENCEVERIFY,
    GPU_OP_NOP4 = 0xb3,
    GPU_OP_NOP5 = 0xb4,
    GPU_OP_NOP6 = 0xb5,
    GPU_OP_NOP7 = 0xb6,
    GPU_OP_NOP8 = 0xb7,
    GPU_OP_NOP9 = 0xb8,
    GPU_OP_NOP10 = 0xb9,

    // Tapscript (BIP 342)
    GPU_OP_CHECKSIGADD = 0xba,

    GPU_OP_INVALIDOPCODE = 0xff
};

// Maximum opcode value
constexpr uint8_t GPU_MAX_OPCODE = GPU_OP_NOP10;

// ============================================================================
// Opcode Classification Helpers
// ============================================================================

// Check if opcode is a push operation (0x00-0x60 or PUSHDATA)
__host__ __device__ inline bool IsOpcodeSmallPush(uint8_t opcode) {
    return opcode <= 0x4b;  // Direct push 0-75 bytes
}

__host__ __device__ inline bool IsOpcodePushData(uint8_t opcode) {
    return opcode == GPU_OP_PUSHDATA1 ||
           opcode == GPU_OP_PUSHDATA2 ||
           opcode == GPU_OP_PUSHDATA4;
}

__host__ __device__ inline bool IsOpcodeSmallInteger(uint8_t opcode) {
    return opcode == GPU_OP_0 ||
           (opcode >= GPU_OP_1 && opcode <= GPU_OP_16);
}

// Check if opcode is disabled (CVE-2010-5137)
__host__ __device__ inline bool IsOpcodeDisabled(uint8_t opcode) {
    return opcode == GPU_OP_CAT ||
           opcode == GPU_OP_SUBSTR ||
           opcode == GPU_OP_LEFT ||
           opcode == GPU_OP_RIGHT ||
           opcode == GPU_OP_INVERT ||
           opcode == GPU_OP_AND ||
           opcode == GPU_OP_OR ||
           opcode == GPU_OP_XOR ||
           opcode == GPU_OP_2MUL ||
           opcode == GPU_OP_2DIV ||
           opcode == GPU_OP_MUL ||
           opcode == GPU_OP_DIV ||
           opcode == GPU_OP_MOD ||
           opcode == GPU_OP_LSHIFT ||
           opcode == GPU_OP_RSHIFT;
}

// Check if opcode is a conditional (IF/NOTIF/ELSE/ENDIF)
__host__ __device__ inline bool IsOpcodeConditional(uint8_t opcode) {
    return opcode >= GPU_OP_IF && opcode <= GPU_OP_ENDIF;
}

// Check if opcode counts toward the opcode limit
__host__ __device__ inline bool IsOpcodeCountable(uint8_t opcode) {
    return opcode > GPU_OP_16;
}

// Check if this is an OP_SUCCESS opcode (Tapscript)
__host__ __device__ inline bool IsOpcodeSuccess(uint8_t opcode) {
    // OP_SUCCESS opcodes: 80, 98, 126-129, 131-134, 137-138, 141-142, 149-153, 187-254
    if (opcode == 0x50) return true;  // 80
    if (opcode == 0x62) return true;  // 98
    if (opcode >= 0x7e && opcode <= 0x81) return true;  // 126-129
    if (opcode >= 0x83 && opcode <= 0x86) return true;  // 131-134
    if (opcode >= 0x89 && opcode <= 0x8a) return true;  // 137-138
    if (opcode >= 0x8d && opcode <= 0x8e) return true;  // 141-142
    if (opcode >= 0x95 && opcode <= 0x99) return true;  // 149-153
    if (opcode >= 0xbb && opcode <= 0xfe) return true;  // 187-254
    return false;
}

// Get the small integer value for OP_0 through OP_16
__host__ __device__ inline int8_t GetSmallIntegerValue(uint8_t opcode) {
    if (opcode == GPU_OP_0) return 0;
    if (opcode >= GPU_OP_1 && opcode <= GPU_OP_16) {
        return static_cast<int8_t>(opcode - GPU_OP_1 + 1);
    }
    if (opcode == GPU_OP_1NEGATE) return -1;
    return 0;  // Invalid
}

// ============================================================================
// Script Parsing Helpers
// ============================================================================

// Get the number of bytes to read for a push operation
// Returns 0 if not a valid push opcode
__host__ __device__ inline uint32_t GetPushDataSize(const uint8_t* script, uint32_t script_len, uint32_t pc) {
    if (pc >= script_len) return 0;

    uint8_t opcode = script[pc];

    // Direct push (1-75 bytes)
    if (opcode >= 0x01 && opcode <= 0x4b) {
        return opcode;
    }

    // OP_PUSHDATA1: next byte is size
    if (opcode == GPU_OP_PUSHDATA1) {
        if (pc + 1 >= script_len) return 0;
        return script[pc + 1];
    }

    // OP_PUSHDATA2: next 2 bytes are size (little-endian)
    if (opcode == GPU_OP_PUSHDATA2) {
        if (pc + 2 >= script_len) return 0;
        return script[pc + 1] | (static_cast<uint32_t>(script[pc + 2]) << 8);
    }

    // OP_PUSHDATA4: next 4 bytes are size (little-endian)
    if (opcode == GPU_OP_PUSHDATA4) {
        if (pc + 4 >= script_len) return 0;
        return script[pc + 1] |
               (static_cast<uint32_t>(script[pc + 2]) << 8) |
               (static_cast<uint32_t>(script[pc + 3]) << 16) |
               (static_cast<uint32_t>(script[pc + 4]) << 24);
    }

    return 0;
}

// Get the total number of bytes consumed by a push operation (opcode + size bytes + data)
__host__ __device__ inline uint32_t GetPushOpcodeSize(const uint8_t* script, uint32_t script_len, uint32_t pc) {
    if (pc >= script_len) return 0;

    uint8_t opcode = script[pc];

    // Direct push (1-75 bytes): 1 byte opcode + N bytes data
    if (opcode >= 0x01 && opcode <= 0x4b) {
        return 1 + opcode;
    }

    // OP_PUSHDATA1: 1 byte opcode + 1 byte size + N bytes data
    if (opcode == GPU_OP_PUSHDATA1) {
        if (pc + 1 >= script_len) return 0;
        return 2 + script[pc + 1];
    }

    // OP_PUSHDATA2: 1 byte opcode + 2 bytes size + N bytes data
    if (opcode == GPU_OP_PUSHDATA2) {
        if (pc + 2 >= script_len) return 0;
        uint32_t size = script[pc + 1] | (static_cast<uint32_t>(script[pc + 2]) << 8);
        return 3 + size;
    }

    // OP_PUSHDATA4: 1 byte opcode + 4 bytes size + N bytes data
    if (opcode == GPU_OP_PUSHDATA4) {
        if (pc + 4 >= script_len) return 0;
        uint32_t size = script[pc + 1] |
                        (static_cast<uint32_t>(script[pc + 2]) << 8) |
                        (static_cast<uint32_t>(script[pc + 3]) << 16) |
                        (static_cast<uint32_t>(script[pc + 4]) << 24);
        return 5 + size;
    }

    // OP_0, OP_1NEGATE, OP_1-OP_16: 1 byte opcode, no data
    if (opcode == GPU_OP_0 || opcode == GPU_OP_1NEGATE ||
        (opcode >= GPU_OP_1 && opcode <= GPU_OP_16)) {
        return 1;
    }

    // Non-push opcode
    return 1;
}

// Check minimal push encoding (BIP 62 rule 3)
__host__ __device__ inline bool CheckMinimalPush(const uint8_t* data, uint32_t size, uint8_t opcode) {
    if (size == 0) {
        // Could have used OP_0
        return opcode == GPU_OP_0;
    } else if (size == 1 && data[0] >= 1 && data[0] <= 16) {
        // Could have used OP_1 through OP_16
        return opcode == (GPU_OP_1 + data[0] - 1);
    } else if (size == 1 && data[0] == 0x81) {
        // Could have used OP_1NEGATE
        return opcode == GPU_OP_1NEGATE;
    } else if (size <= 75) {
        // Could have used direct push
        return opcode == size;
    } else if (size <= 255) {
        // Could have used OP_PUSHDATA1
        return opcode == GPU_OP_PUSHDATA1;
    } else if (size <= 65535) {
        // Could have used OP_PUSHDATA2
        return opcode == GPU_OP_PUSHDATA2;
    }
    return true;  // OP_PUSHDATA4 is the only option
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_OPCODES_CUH
