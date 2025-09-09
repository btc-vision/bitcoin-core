#include "gpu_test.h"
#include "gpu_logging.h"
#include <logging.h>
#include <cstdio>
#include <string>

// C-linkage functions for CUDA interop
extern "C" {

void LogGPUDebug(const char* message) {
    kernel::LogGPUDebug(message);
}

void LogGPUInfo(const char* message) {
    kernel::LogGPUInfo(message);
}

void LogGPUInfoFormatted(const char* format, int value) {
    kernel::LogGPUInfoFormatted(format, value);
}

} // extern "C"

namespace kernel {

void LogGPUInfo(const char* message) {
    // Remove trailing newline if present
    std::string msg(message);
    if (!msg.empty() && msg.back() == '\n') {
        msg.pop_back();
    }
    LogInfo("%s", msg);
}

void LogGPUDebug(const char* message) {
    // Remove trailing newline if present
    std::string msg(message);
    if (!msg.empty() && msg.back() == '\n') {
        msg.pop_back();
    }
    LogDebug(BCLog::GPU, "%s", msg);
}

void LogGPUWarning(const char* message) {
    // Remove trailing newline if present
    std::string msg(message);
    if (!msg.empty() && msg.back() == '\n') {
        msg.pop_back();
    }
    LogWarning("%s", msg);
}

void LogGPUInfoFormatted(const char* format, int value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfo(buffer);
}

void LogGPUInfoFormattedStr(const char* format, const char* value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfo(buffer);
}

void LogGPUInfoFormattedMulti(const char* format, int value1, const char* value2) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value1, value2);
    LogGPUInfo(buffer);
}

void LogGPUInfoFormattedLong(const char* format, unsigned long long value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfo(buffer);
}

void LogGPUInfoFormattedFloat(const char* format, float value) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, value);
    LogGPUInfo(buffer);
}

void LogGPUInfoFormattedThree(const char* format, int v1, int v2, int v3) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, v1, v2, v3);
    LogGPUInfo(buffer);
}

} // namespace kernel