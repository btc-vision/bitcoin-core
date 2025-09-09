#ifndef BITCOIN_GPU_KERNEL_GPU_VALIDATION_H
#define BITCOIN_GPU_KERNEL_GPU_VALIDATION_H

#include "gpu_types.h"
#include <cstdint>
#include <cstddef>

namespace gpu {

// Script validation functions (host-callable for testing)
bool ValidateP2PKHScript(const uint8_t* script, size_t size);
bool ValidateP2WPKHScript(const uint8_t* script, size_t size);

// Hash functions for testing
void sha256(const uint8_t* data, size_t len, uint8_t* out);
void ripemd160(const uint8_t* data, size_t len, uint8_t* out);

// Script type identification
ScriptType IdentifyScriptType(const uint8_t* script, size_t size);

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_VALIDATION_H