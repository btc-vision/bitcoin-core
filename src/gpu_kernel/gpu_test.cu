#include "gpu_test.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <cstdio>

__global__ void testKernel(int* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] = d_data[idx] * 2 + 1;
    }
}

namespace kernel {

bool TestGPUKernel() {
    const int size = 1024;
    const int bytes = size * sizeof(int);
    
    int* h_data = new int[size];
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }
    
    int* d_data;
    cudaError_t err = cudaMalloc(&d_data, bytes);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        delete[] h_data;
        return false;
    }
    
    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        cudaFree(d_data);
        delete[] h_data;
        return false;
    }
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    testKernel<<<gridSize, blockSize>>>(d_data, size);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        cudaFree(d_data);
        delete[] h_data;
        return false;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA device synchronize failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        cudaFree(d_data);
        delete[] h_data;
        return false;
    }
    
    int* h_result = new int[size];
    err = cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA memcpy from device failed: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        cudaFree(d_data);
        delete[] h_data;
        delete[] h_result;
        return false;
    }
    
    bool success = true;
    for (int i = 0; i < size && i < 10; i++) {
        int expected = i * 2 + 1;
        if (h_result[i] != expected) {
            char msg[256];
            snprintf(msg, sizeof(msg), "GPU test failed at index %d: expected %d, got %d\n", i, expected, h_result[i]);
            LogGPUDebug(msg);
            success = false;
        }
    }
    
    if (success) {
        LogGPUDebug("GPU test kernel executed successfully!\n");
    }
    
    cudaFree(d_data);
    delete[] h_data;
    delete[] h_result;
    
    return success;
}

void PrintGPUInfo() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        LogGPUDebug(msg);
        return;
    }
    
    if (deviceCount == 0) {
        LogGPUInfo("No CUDA-capable GPU devices found.\n");
        return;
    }
    
    LogGPUInfoFormatted("Found %d CUDA GPU device(s):\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err == cudaSuccess) {
            char msg[256];
            snprintf(msg, sizeof(msg), "  GPU Device %d: %s\n", i, prop.name);
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    Compute capability: %d.%d\n", prop.major, prop.minor);
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    Total memory: %llu MB\n", (unsigned long long)(prop.totalGlobalMem / (1024 * 1024)));
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    Multiprocessors: %d\n", prop.multiProcessorCount);
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    Max threads per block: %d\n", prop.maxThreadsPerBlock);
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    Max grid dimensions: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    Warp size: %d\n", prop.warpSize);
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    L2 cache size: %d KB\n", prop.l2CacheSize / 1024);
            LogGPUInfo(msg);
            snprintf(msg, sizeof(msg), "    Memory bus width: %d bits\n", prop.memoryBusWidth);
            LogGPUInfo(msg);
        }
    }
}

} // namespace kernel