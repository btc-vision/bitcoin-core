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

// Script type identification
ScriptType IdentifyScriptType(const uint8_t* script, size_t size) {
    if (size == 25 && script[0] == 0x76 && script[1] == 0xa9 && script[2] == 0x14) {
        return SCRIPT_TYPE_P2PKH;  // OP_DUP OP_HASH160 <20 bytes>
    }
    if (size == 22 && script[0] == 0x00 && script[1] == 0x14) {
        return SCRIPT_TYPE_P2WPKH;  // OP_0 <20 bytes>
    }
    if (size == 23 && script[0] == 0xa9 && script[1] == 0x14) {
        return SCRIPT_TYPE_P2SH;  // OP_HASH160 <20 bytes>
    }
    if (size == 34 && script[0] == 0x00 && script[1] == 0x20) {
        return SCRIPT_TYPE_P2WSH;  // OP_0 <32 bytes>
    }
    if (size == 34 && script[0] == 0x51 && script[1] == 0x20) {
        return SCRIPT_TYPE_P2TR;  // OP_1 <32 bytes>
    }
    return SCRIPT_TYPE_UNKNOWN;
}

} // namespace gpu