// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// This file exists solely to force CMake to perform CUDA device linking
// for the bitcoind executable. Without at least one .cu file in the
// executable, CMake won't perform device linking and the CUDA kernels
// from bitcoin_kernel_gpu won't be properly registered.

#include <cuda_runtime.h>

namespace {
// Empty namespace - no actual code needed
// The mere presence of this .cu file triggers CUDA device linking
}
