// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_LOGGING_H
#define BITCOIN_GPU_KERNEL_GPU_LOGGING_H

#ifdef __cplusplus
extern "C" {
#endif

void LogGPUDebug(const char* message);
void LogGPUInfo(const char* message);
void LogGPUInfoFormatted(const char* format, int value);

#ifdef __cplusplus
}
#endif

#endif // BITCOIN_GPU_KERNEL_GPU_LOGGING_H