// Copyright (c) 2024-present The Bitcoin Core developers
// GPU UTXO Initialization Tests - Complete coverage for GPU initialization

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_utxo.h>
#include <gpu_kernel/gpu_logging.h>
#include <cuda_runtime.h>
#include <random.h>

BOOST_AUTO_TEST_SUITE(gpu_initialization_tests)

BOOST_FIXTURE_TEST_CASE(gpu_device_detection, BasicTestingSetup)
{
    // Test CUDA device detection
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        BOOST_TEST_MESSAGE("No CUDA devices found - skipping GPU tests");
        return;
    }
    
    BOOST_CHECK_GE(deviceCount, 1);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    BOOST_TEST_MESSAGE("GPU Device: " + std::string(prop.name));
    BOOST_TEST_MESSAGE("Total VRAM: " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB");
    BOOST_TEST_MESSAGE("Compute Capability: " + std::to_string(prop.major) + "." + std::to_string(prop.minor));
    
    // Verify minimum requirements
    BOOST_CHECK_GE(prop.major, 6); // Require at least Pascal architecture
    BOOST_CHECK_GE(prop.totalGlobalMem, 1ULL * 1024 * 1024 * 1024); // At least 1GB VRAM
}

BOOST_FIXTURE_TEST_CASE(gpu_memory_allocation_default, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    
    // Initialize with default (95% of available VRAM)
    bool result = utxoSet.Initialize();
    BOOST_CHECK(result);
    
    // Verify allocations
    BOOST_CHECK_GT(utxoSet.GetMaxUTXOs(), 0);
    BOOST_CHECK_GT(utxoSet.GetMaxVRAMLimit(), 0);
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 0);
    
    // Check that we're using less than 95% of available VRAM
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    BOOST_CHECK_LE(used_mem, static_cast<size_t>(total_mem * 0.95));
}

BOOST_FIXTURE_TEST_CASE(gpu_memory_allocation_custom_limits, BasicTestingSetup)
{
    // Test with various memory limits
    std::vector<size_t> limits = {
        100 * 1024 * 1024,      // 100MB
        500 * 1024 * 1024,      // 500MB
        1024 * 1024 * 1024,     // 1GB
        2048 * 1024 * 1024      // 2GB
    };
    
    for (size_t limit : limits) {
        gpu::GPUUTXOSet utxoSet;
        bool result = utxoSet.Initialize(limit);
        
        if (result) {
            // Verify the limit was respected
            BOOST_CHECK_LE(utxoSet.GetTotalVRAMUsed(), limit);
            BOOST_CHECK_LE(utxoSet.GetMaxVRAMLimit(), limit);
            
            // Verify proportional scaling
            size_t expectedUTXOs = limit / (sizeof(gpu::UTXOHeader) + 100); // Rough estimate
            BOOST_CHECK_LE(utxoSet.GetMaxUTXOs(), expectedUTXOs * 2); // Allow some variance
        }
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_memory_allocation_failure, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    
    // Try to allocate way too much memory (100TB)
    bool result = utxoSet.Initialize(100ULL * 1024 * 1024 * 1024 * 1024);
    BOOST_CHECK(!result);
    
    // Verify nothing was allocated
    BOOST_CHECK_EQUAL(utxoSet.GetMaxUTXOs(), 0);
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 0);
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_table_initialization, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Verify all 4 hash tables are initialized
    for (int i = 0; i < 4; i++) {
        BOOST_CHECK(utxoSet.GetHashTable(i) != nullptr);
    }
    
    // Verify hash tables are cleared (all entries should be 0xFFFFFFFF)
    std::vector<uint32_t> sample(1000);
    for (int i = 0; i < 4; i++) {
        cudaMemcpy(sample.data(), utxoSet.GetHashTable(i), 
                   sample.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        for (uint32_t val : sample) {
            BOOST_CHECK_EQUAL(val, 0xFFFFFFFF);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_memory_info_reporting, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Check initial state
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 0);
    BOOST_CHECK_EQUAL(utxoSet.GetLoadFactor(), 0);
    BOOST_CHECK_EQUAL(utxoSet.GetTotalFreeSpace(), 0);
    BOOST_CHECK_GT(utxoSet.GetTotalVRAMUsed(), 0);
    
    // Add some UTXOs and verify counters update
    for (int i = 0; i < 100; i++) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.amount = i * COIN;
        header.txid_index = i;
        header.vout = 0;
        header.script_size = 25;
        
        utxoSet.AddUTXO(txid, 0, header, nullptr);
    }
    
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 100);
    BOOST_CHECK_GT(utxoSet.GetLoadFactor(), 0);
}

BOOST_FIXTURE_TEST_CASE(gpu_reinitialization, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    
    // First initialization
    BOOST_CHECK(utxoSet.Initialize(100 * 1024 * 1024));
    size_t firstMaxUTXOs = utxoSet.GetMaxUTXOs();
    
    // Add some data
    for (int i = 0; i < 10; i++) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = i;
        utxoSet.AddUTXO(txid, 0, header, nullptr);
    }
    
    // Reinitialize with different size
    BOOST_CHECK(utxoSet.Initialize(200 * 1024 * 1024));
    
    // Verify clean state after reinitialization
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 0);
    BOOST_CHECK_NE(utxoSet.GetMaxUTXOs(), firstMaxUTXOs);
}

BOOST_FIXTURE_TEST_CASE(gpu_cuda_error_handling, BasicTestingSetup)
{
    // Test CUDA error recovery
    gpu::GPUUTXOSet utxoSet;
    
    // Force a CUDA error by setting invalid device
    cudaSetDevice(999);
    
    // Initialization should fail gracefully
    bool result = utxoSet.Initialize();
    BOOST_CHECK(!result);
    
    // Reset to valid device
    cudaSetDevice(0);
    
    // Should work now
    gpu::GPUUTXOSet utxoSet2;
    result = utxoSet2.Initialize();
    BOOST_CHECK(result);
}

BOOST_FIXTURE_TEST_CASE(gpu_memory_fragmentation_tracking, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Initially no fragmentation
    BOOST_CHECK_EQUAL(utxoSet.GetTotalFreeSpace(), 0);
    
    // Add UTXOs with scripts
    std::vector<uint256> txids;
    for (int i = 0; i < 100; i++) {
        uint256 txid = GetRandHash();
        txids.push_back(txid);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = 25 + (i % 10) * 10; // Variable sizes
        header.txid_index = i;
        
        std::vector<uint8_t> script(header.script_size, 0xAB);
        utxoSet.AddUTXO(txid, 0, header, script.data());
    }
    
    size_t initialScriptBlobUsed = utxoSet.GetScriptBlobUsed();
    
    // Spend some to create fragmentation
    for (int i = 0; i < 50; i++) {
        utxoSet.SpendUTXO(txids[i], 0);
    }
    
    // Free space should have increased
    BOOST_CHECK_GT(utxoSet.GetTotalFreeSpace(), 0);
    
    // But script blob used should remain the same (fragmented)
    BOOST_CHECK_EQUAL(utxoSet.GetScriptBlobUsed(), initialScriptBlobUsed);
}

BOOST_FIXTURE_TEST_CASE(gpu_parallel_initialization, BasicTestingSetup)
{
    // Test that multiple UTXO sets can coexist
    gpu::GPUUTXOSet utxoSet1, utxoSet2;
    
    // Both should initialize successfully if enough VRAM
    bool result1 = utxoSet1.Initialize(50 * 1024 * 1024);
    bool result2 = utxoSet2.Initialize(50 * 1024 * 1024);
    
    if (result1 && result2) {
        // Both initialized - verify independence
        uint256 txid1 = GetRandHash();
        uint256 txid2 = GetRandHash();
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        
        utxoSet1.AddUTXO(txid1, 0, header, nullptr);
        utxoSet2.AddUTXO(txid2, 0, header, nullptr);
        
        BOOST_CHECK(utxoSet1.HasUTXO(txid1, 0));
        BOOST_CHECK(!utxoSet1.HasUTXO(txid2, 0));
        BOOST_CHECK(utxoSet2.HasUTXO(txid2, 0));
        BOOST_CHECK(!utxoSet2.HasUTXO(txid1, 0));
    }
}

BOOST_AUTO_TEST_SUITE_END()