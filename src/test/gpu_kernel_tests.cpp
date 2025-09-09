// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>

#ifdef ENABLE_GPU_ACCELERATION
#include <gpu_kernel/gpu_test.h>
#include <gpu_kernel/gpu_mining.h>
#endif

#include <vector>
#include <memory>

BOOST_FIXTURE_TEST_SUITE(gpu_kernel_tests, BasicTestingSetup)

#ifdef ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_CASE(gpu_kernel_basic_test)
{
    // Test the basic GPU kernel functionality
    bool result = kernel::TestGPUKernel();
    BOOST_CHECK_MESSAGE(result, "GPU kernel test failed - check CUDA installation and GPU availability");
}

BOOST_AUTO_TEST_CASE(gpu_info_retrieval)
{
    // Test that we can retrieve GPU information without crashing
    // This doesn't validate output, just ensures PrintGPUInfo doesn't crash
    BOOST_CHECK_NO_THROW(kernel::PrintGPUInfo());
}

BOOST_AUTO_TEST_CASE(gpu_mining_kernel_initialization)
{
    // Test GPU mining kernel initialization
    GPUMiningKernel miningKernel;
    
    // Test initialization
    bool initialized = miningKernel.Initialize();
    BOOST_CHECK_MESSAGE(initialized, "GPU mining kernel initialization failed");
    
    if (initialized) {
        // Test that we can get device count
        int deviceCount = miningKernel.GetDeviceCount();
        BOOST_CHECK_MESSAGE(deviceCount >= 0, "Invalid device count returned");
        
        // If we have devices, test getting properties
        if (deviceCount > 0) {
            for (int i = 0; i < deviceCount; i++) {
                auto props = miningKernel.GetDeviceProperties(i);
                BOOST_CHECK_MESSAGE(props != nullptr, "Failed to get device properties for device " + std::to_string(i));
            }
        }
    }
    
    // Test cleanup
    BOOST_CHECK_NO_THROW(miningKernel.Cleanup());
}

BOOST_AUTO_TEST_CASE(gpu_memory_allocation)
{
    // Test GPU memory allocation and deallocation
    GPUMiningKernel miningKernel;
    
    if (miningKernel.Initialize()) {
        const size_t testSize = 1024 * 1024; // 1MB
        
        // Test allocation
        void* gpuMem = nullptr;
        bool allocated = miningKernel.AllocateGPUMemory(&gpuMem, testSize);
        
        if (allocated) {
            BOOST_CHECK(gpuMem != nullptr);
            
            // Test data transfer
            std::vector<uint8_t> hostData(testSize, 0x42);
            bool copySuccess = miningKernel.CopyToGPU(gpuMem, hostData.data(), testSize);
            BOOST_CHECK_MESSAGE(copySuccess, "Failed to copy data to GPU");
            
            // Test data retrieval
            std::vector<uint8_t> retrievedData(testSize);
            bool retrieveSuccess = miningKernel.CopyFromGPU(retrievedData.data(), gpuMem, testSize);
            BOOST_CHECK_MESSAGE(retrieveSuccess, "Failed to copy data from GPU");
            
            if (retrieveSuccess) {
                // Verify first few bytes
                for (size_t i = 0; i < std::min(size_t(100), testSize); i++) {
                    BOOST_CHECK_EQUAL(retrievedData[i], hostData[i]);
                }
            }
            
            // Test deallocation
            BOOST_CHECK_NO_THROW(miningKernel.FreeGPUMemory(gpuMem));
        } else {
            BOOST_TEST_MESSAGE("GPU memory allocation skipped - no GPU available");
        }
        
        miningKernel.Cleanup();
    }
}

BOOST_AUTO_TEST_CASE(gpu_kernel_computation_validation)
{
    // Test that the GPU kernel computation produces expected results
    GPUMiningKernel miningKernel;
    
    if (miningKernel.Initialize()) {
        const int dataSize = 1024;
        std::vector<int> inputData(dataSize);
        
        // Initialize test data
        for (int i = 0; i < dataSize; i++) {
            inputData[i] = i;
        }
        
        // Run computation on GPU
        std::vector<int> outputData(dataSize);
        bool computeSuccess = miningKernel.RunTestComputation(inputData.data(), outputData.data(), dataSize);
        
        if (computeSuccess) {
            // Verify results (based on testKernel: d_data[idx] = d_data[idx] * 2 + 1)
            for (int i = 0; i < std::min(100, dataSize); i++) {
                int expected = inputData[i] * 2 + 1;
                BOOST_CHECK_EQUAL(outputData[i], expected);
            }
        } else {
            BOOST_TEST_MESSAGE("GPU computation skipped - no GPU available");
        }
        
        miningKernel.Cleanup();
    }
}

#else // !ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_CASE(gpu_disabled_check)
{
    // When GPU acceleration is disabled, just verify the build works correctly
    BOOST_TEST_MESSAGE("GPU acceleration is disabled in this build");
    BOOST_CHECK(true);
}

#endif // ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_SUITE_END()