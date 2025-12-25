// C++ wrapper for GPU UTXO functions that need Bitcoin Core headers
#include "gpu_utxo.h"
#include "gpu_utils.h"
#include "gpu_logging.h"
#include <coins.h>
#include <primitives/transaction.h>
#include <script/script.h>
#include <validation.h>
#include <chrono>

namespace gpu {

// Note: The LoadFromCPU implementation is in gpu_utxo_loader.cu
// This file just provides helper functions that need Bitcoin Core headers

// Script type identification - matches Bitcoin Core's TxoutType detection
ScriptType IdentifyScriptType(const uint8_t* script, size_t size) {
    if (!script || size == 0) return SCRIPT_TYPE_UNKNOWN;

    // P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
    if (size == 25 && script[0] == 0x76 && script[1] == 0xa9 &&
        script[2] == 0x14 && script[23] == 0x88 && script[24] == 0xac) {
        return SCRIPT_TYPE_P2PKH;
    }

    // P2SH: OP_HASH160 <20 bytes> OP_EQUAL
    if (size == 23 && script[0] == 0xa9 && script[1] == 0x14 && script[22] == 0x87) {
        return SCRIPT_TYPE_P2SH;
    }

    // P2WPKH: OP_0 <20 bytes>
    if (size == 22 && script[0] == 0x00 && script[1] == 0x14) {
        return SCRIPT_TYPE_P2WPKH;
    }

    // P2WSH: OP_0 <32 bytes>
    if (size == 34 && script[0] == 0x00 && script[1] == 0x20) {
        return SCRIPT_TYPE_P2WSH;
    }

    // P2TR: OP_1 <32 bytes>
    if (size == 34 && script[0] == 0x51 && script[1] == 0x20) {
        return SCRIPT_TYPE_P2TR;
    }

    // P2PK: <33 or 65 byte pubkey> OP_CHECKSIG
    if ((size == 35 && script[0] == 0x21 && script[34] == 0xac &&
         (script[1] == 0x02 || script[1] == 0x03)) ||
        (size == 67 && script[0] == 0x41 && script[66] == 0xac &&
         script[1] == 0x04)) {
        return SCRIPT_TYPE_P2PK;
    }

    // NULL_DATA: OP_RETURN <optional data>
    if (size >= 1 && script[0] == 0x6a) {
        return SCRIPT_TYPE_NULL_DATA;
    }

    // WITNESS_UNKNOWN: OP_N <2-40 bytes> where N is 2-16
    if (size >= 4 && size <= 42 && script[0] >= 0x52 && script[0] <= 0x60) {
        uint8_t push_size = script[1];
        if (push_size >= 2 && push_size <= 40 && size == (size_t)(2 + push_size)) {
            return SCRIPT_TYPE_WITNESS_UNKNOWN;
        }
    }

    // MULTISIG: OP_M <pubkeys> OP_N OP_CHECKMULTISIG
    if (size >= 37 && (script[size-1] == 0xae || script[size-1] == 0xaf)) {
        if (script[0] >= 0x51 && script[0] <= 0x60 &&
            script[size-2] >= 0x51 && script[size-2] <= 0x60) {
            return SCRIPT_TYPE_MULTISIG;
        }
    }

    return SCRIPT_TYPE_NONSTANDARD;
}

} // namespace gpu