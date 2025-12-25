// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "gpu_logging.h"
#include <logging.h>
#include <cstdio>
#include <string>

namespace {

void LogGPUInfoInternal(const char* message) {
    std::string msg(message);
    if (!msg.empty() && msg.back() == '\n') {
        msg.pop_back();
    }
    LogInfo("%s", msg);
}

void LogGPUDebugInternal(const char* message) {
    std::string msg(message);
    if (!msg.empty() && msg.back() == '\n') {
        msg.pop_back();
    }
    LogDebug(BCLog::GPU, "%s", msg);
}

} // anonymous namespace

// C-linkage functions for CUDA interop
extern "C" {

void LogGPUDebug(const char* message) {
    LogGPUDebugInternal(message);
}

void LogGPUInfo(const char* message) {
    LogGPUInfoInternal(message);
}

void LogGPUInfoFormatted(const char* format, int value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfoInternal(buffer);
}

} // extern "C"

namespace kernel {

void LogGPUInfo(const char* message) {
    LogGPUInfoInternal(message);
}

void LogGPUDebug(const char* message) {
    LogGPUDebugInternal(message);
}

void LogGPUWarning(const char* message) {
    std::string msg(message);
    if (!msg.empty() && msg.back() == '\n') {
        msg.pop_back();
    }
    LogWarning("%s", msg);
}

void LogGPUInfoFormatted(const char* format, int value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfoInternal(buffer);
}

void LogGPUInfoFormattedStr(const char* format, const char* value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfoInternal(buffer);
}

void LogGPUInfoFormattedMulti(const char* format, int value1, const char* value2) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value1, value2);
    LogGPUInfoInternal(buffer);
}

void LogGPUInfoFormattedLong(const char* format, unsigned long long value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfoInternal(buffer);
}

void LogGPUInfoFormattedFloat(const char* format, float value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfoInternal(buffer);
}

void LogGPUInfoFormattedThree(const char* format, int v1, int v2, int v3) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, v1, v2, v3);
    LogGPUInfoInternal(buffer);
}

} // namespace kernel
