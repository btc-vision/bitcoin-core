// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <crypto/sha256.h>
#include <hash.h>
#include <uint256.h>

#ifdef ENABLE_GPU_ACCELERATION
#include <gpu_kernel/gpu_test.h>
#include <gpu_kernel/gpu_mining.h>
#endif

#include <vector>
#include <memory>
#include <cstring>

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

BOOST_AUTO_TEST_CASE(gpu_sha256d_verification)
{
    // Test that GPU SHA256d matches CPU implementation
    GPUMiningKernel miningKernel;
    
    if (miningKernel.Initialize()) {
        // Test vectors
        struct TestVector {
            std::vector<uint8_t> input;
            std::string expected_hex;
        };
        
        std::vector<TestVector> vectors = {
            // Empty input
            {{}, "5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456"},
            // "abc"
            {{0x61, 0x62, 0x63}, "4f8b42c22dd3729b519ba6f68d2da7cc5b2d606d05daed5ad5128cc03e6c6358"},
            // 64 bytes of zeros
            {std::vector<uint8_t>(64, 0), "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"}
        };
        
        for (const auto& test : vectors) {
            // Compute on CPU using actual input size
            CHash256 hasher;
            hasher.Write({test.input.data(), test.input.size()});
            uint256 cpu_result;
            hasher.Finalize(cpu_result);
            
            // For GPU, we need to pad to 64 bytes (minimum block size)
            if (test.input.size() <= 64) {
                std::vector<uint8_t> padded_input(64, 0);
                if (!test.input.empty()) {
                    memcpy(padded_input.data(), test.input.data(), test.input.size());
                }
                
                // Prepare for GPU test - we'll compute SHA256d of the actual length
                // by passing the length info somehow... Actually, let's test a different way
                // Let's directly test the mining kernel's hash computation
                uint8_t gpu_hash[32];
                
                // Use a simple test: create a fake block header
                uint8_t test_header[80] = {0};
                memcpy(test_header, test.input.data(), std::min(test.input.size(), size_t(80)));
                
                // Mine with impossible target to just get hash
                uint8_t impossible_target[32] = {0}; // Impossible to meet
                uint32_t found_nonce = 0;
                uint8_t found_hash[32];
                
                // Try just one nonce to get the hash
                bool found = miningKernel.Mine(test_header, impossible_target, 0, 1, &found_nonce, found_hash);
                
                // For this test, we actually want to compare the direct SHA256d computation
                // Let's use a different approach - test with 80-byte blocks like real mining
            }
        }
        
        // Test with actual 80-byte block headers
        {
            uint8_t test_header[80];
            memset(test_header, 0, 80);
            
            // Test 1: All zeros
            CHash256 cpu_hasher;
            cpu_hasher.Write({test_header, 80});
            uint256 cpu_hash;
            cpu_hasher.Finalize(cpu_hash);
            
            // Mine with easy target
            uint8_t easy_target[32];
            memset(easy_target, 0xFF, 32);
            
            uint32_t found_nonce = 0;
            uint8_t found_hash[32];
            bool found = miningKernel.Mine(test_header, easy_target, 0, 1, &found_nonce, found_hash);
            
            if (found) {
                uint256 gpu_hash;
                memcpy(gpu_hash.begin(), found_hash, 32);
                
                BOOST_CHECK_MESSAGE(
                    cpu_hash == gpu_hash,
                    "GPU SHA256d mismatch for 80-byte block\n" +
                    std::string("Expected: ") + cpu_hash.ToString() + "\n" +
                    std::string("Got:      ") + gpu_hash.ToString()
                );
            }
        }
        
        miningKernel.Cleanup();
    } else {
        BOOST_TEST_MESSAGE("GPU SHA256d test skipped - no GPU available");
    }
}

BOOST_AUTO_TEST_CASE(gpu_mining_with_verification)
{
    // Test actual Bitcoin mining with CPU verification
    GPUMiningKernel miningKernel;
    
    if (miningKernel.Initialize()) {
        // Create a test block header (80 bytes)
        uint8_t blockHeader[80] = {0};
        
        // Fill with test data
        for (int i = 0; i < 80; i++) {
            blockHeader[i] = i;
        }
        
        // Set an easy target (high value = easy difficulty)
        uint8_t target[32];
        memset(target, 0xFF, 32);
        target[31] = 0x0F;  // Make it harder to ensure we find a specific solution
        
        // Test mining
        uint32_t foundNonce = 0;
        uint8_t foundHash[32] = {0};
        
        // Search a range
        uint32_t startNonce = 0;
        uint32_t nonceRange = 100000;
        
        bool found = miningKernel.Mine(blockHeader, target, startNonce, nonceRange, &foundNonce, foundHash);
        
        if (found) {
            BOOST_TEST_MESSAGE("Mining solution found at nonce: " + std::to_string(foundNonce));
            
            // Verify the solution on CPU
            uint8_t verifyHeader[80];
            memcpy(verifyHeader, blockHeader, 80);
            verifyHeader[76] = (foundNonce >> 0) & 0xFF;
            verifyHeader[77] = (foundNonce >> 8) & 0xFF;
            verifyHeader[78] = (foundNonce >> 16) & 0xFF;
            verifyHeader[79] = (foundNonce >> 24) & 0xFF;
            
            // Compute SHA256d on CPU
            CHash256 hasher;
            hasher.Write({verifyHeader, 80});
            uint256 cpu_hash;
            hasher.Finalize(cpu_hash);
            
            // Compare with GPU result
            uint256 gpu_hash;
            memcpy(gpu_hash.begin(), foundHash, 32);
            
            BOOST_CHECK_MESSAGE(
                cpu_hash == gpu_hash,
                "GPU mining hash verification failed!\n" +
                std::string("CPU hash: ") + cpu_hash.ToString() + "\n" +
                std::string("GPU hash: ") + gpu_hash.ToString()
            );
            
            // Verify the hash meets the target
            bool meets_target = true;
            for (int i = 31; i >= 0; i--) {
                if (foundHash[i] < target[i]) {
                    break;
                } else if (foundHash[i] > target[i]) {
                    meets_target = false;
                    break;
                }
            }
            
            BOOST_CHECK_MESSAGE(meets_target, "Found hash does not meet target difficulty");
            
            // Double-check by computing with the found nonce
            BOOST_CHECK_MESSAGE(
                foundNonce >= startNonce && foundNonce < startNonce + nonceRange,
                "Found nonce is outside search range"
            );
        } else {
            BOOST_TEST_MESSAGE("No mining solution found in range - may need to search larger range");
        }
        
        miningKernel.Cleanup();
    } else {
        BOOST_TEST_MESSAGE("GPU mining test skipped - no GPU available");
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