#ifndef BITCOIN_KERNEL_GPU_TEST_H
#define BITCOIN_KERNEL_GPU_TEST_H

namespace kernel {

bool TestGPUKernel();
void PrintGPUInfo();

// Logging wrapper functions to avoid including logging.h in CUDA files
void LogGPUInfo(const char* message);
void LogGPUDebug(const char* message);
void LogGPUWarning(const char* message);
void LogGPUInfoFormatted(const char* format, int value);
void LogGPUInfoFormattedStr(const char* format, const char* value);
void LogGPUInfoFormattedMulti(const char* format, int value1, const char* value2);
void LogGPUInfoFormattedLong(const char* format, unsigned long long value);
void LogGPUInfoFormattedFloat(const char* format, float value);
void LogGPUInfoFormattedThree(const char* format, int v1, int v2, int v3);

} // namespace kernel

#endif // BITCOIN_KERNEL_GPU_TEST_H