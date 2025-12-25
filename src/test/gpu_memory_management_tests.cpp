// Copyright (c) 2024-present The Bitcoin Core developers
// GPU Memory Management Tests - Complete coverage for VRAM management and fragmentation

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_utxo.h>
#include <random.h>
#include <cuda_runtime.h>

BOOST_AUTO_TEST_SUITE(gpu_memory_management_tests)

BOOST_FIXTURE_TEST_CASE(memory_95_percent_limit, BasicTestingSetup)
{
    // Verify 95% VRAM limit is respected
    size_t free_before, total_before;
    cudaMemGetInfo(&free_before, &total_before);
    
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    
    size_t used = free_before - free_after;
    
    // Should use less than 95% of what was available
    BOOST_CHECK_LE(used, static_cast<size_t>(free_before * 0.95));
    
    // Should use at least some memory
    BOOST_CHECK_GT(used, 0);
}

BOOST_FIXTURE_TEST_CASE(memory_script_blob_allocation, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    size_t initial_blob_used = utxoSet.GetScriptBlobUsed();
    BOOST_CHECK_EQUAL(initial_blob_used, 0);
    
    // Add UTXOs with different script sizes
    std::vector<size_t> script_sizes = {25, 34, 22, 23, 100, 500, 1000};
    size_t total_script_size = 0;
    
    for (size_t i = 0; i < script_sizes.size(); i++) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = script_sizes[i];
        header.txid_index = i;
        
        std::vector<uint8_t> script(script_sizes[i], 0xAB);
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
        
        total_script_size += script_sizes[i];
    }
    
    // Verify script blob usage matches
    BOOST_CHECK_EQUAL(utxoSet.GetScriptBlobUsed(), total_script_size);
}

BOOST_FIXTURE_TEST_CASE(memory_fragmentation_tracking, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add many UTXOs with scripts
    std::vector<uint256> txids;
    const int NUM_UTXOS = 1000;
    size_t total_script_size = 0;
    
    for (int i = 0; i < NUM_UTXOS; i++) {
        uint256 txid = GetRandHash();
        txids.push_back(txid);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = 25 + (i % 100); // Variable sizes
        header.txid_index = i;
        
        std::vector<uint8_t> script(header.script_size, i % 256);
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
        
        total_script_size += header.script_size;
    }
    
    BOOST_CHECK_EQUAL(utxoSet.GetScriptBlobUsed(), total_script_size);
    BOOST_CHECK_EQUAL(utxoSet.GetTotalFreeSpace(), 0);
    
    // Spend half to create fragmentation
    size_t freed_space = 0;
    for (int i = 0; i < NUM_UTXOS / 2; i++) {
        freed_space += 25 + (i % 100);
        BOOST_CHECK(utxoSet.SpendUTXO(txids[i], 0));
    }
    
    // Free space should match what we freed
    BOOST_CHECK_EQUAL(utxoSet.GetTotalFreeSpace(), freed_space);
    
    // Script blob used should not change (fragmented)
    BOOST_CHECK_EQUAL(utxoSet.GetScriptBlobUsed(), total_script_size);
}

BOOST_FIXTURE_TEST_CASE(memory_txid_deduplication, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add multiple outputs from same transaction
    uint256 txid = GetRandHash();
    const int NUM_OUTPUTS = 100;
    
    for (int vout = 0; vout < NUM_OUTPUTS; vout++) {
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = 0; // Should reuse same txid entry
        header.vout = vout;
        
        BOOST_CHECK(utxoSet.AddUTXO(txid, vout, header, nullptr));
    }
    
    // Txid table should have grown by only 1
    BOOST_CHECK_LE(utxoSet.GetTxidTableUsed(), 1);
    
    // All outputs should be findable
    for (int vout = 0; vout < NUM_OUTPUTS; vout++) {
        BOOST_CHECK(utxoSet.HasUTXO(txid, vout));
    }
}

BOOST_FIXTURE_TEST_CASE(memory_scaling_with_limits, BasicTestingSetup)
{
    // Test different memory limits
    struct TestCase {
        size_t limit;
        size_t expected_min_utxos;
    };
    
    std::vector<TestCase> test_cases = {
        {50 * 1024 * 1024, 100000},      // 50MB -> ~100k UTXOs
        {100 * 1024 * 1024, 200000},     // 100MB -> ~200k UTXOs
        {500 * 1024 * 1024, 1000000},    // 500MB -> ~1M UTXOs
    };
    
    for (const auto& test : test_cases) {
        gpu::GPUUTXOSet utxoSet;
        
        if (utxoSet.Initialize(test.limit)) {
            BOOST_CHECK_GE(utxoSet.GetMaxUTXOs(), test.expected_min_utxos);
            BOOST_CHECK_LE(utxoSet.GetTotalVRAMUsed(), test.limit);
            
            BOOST_TEST_MESSAGE("Limit: " + std::to_string(test.limit / (1024*1024)) + 
                             "MB -> Max UTXOs: " + std::to_string(utxoSet.GetMaxUTXOs()));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(memory_overflow_protection, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize(10 * 1024 * 1024)); // Small 10MB limit
    
    // Try to add more than capacity
    int successful = 0;
    int failed = 0;
    
    for (int i = 0; i < 1000000; i++) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = i;
        
        if (utxoSet.AddUTXO(txid, 0, header, nullptr)) {
            successful++;
        } else {
            failed++;
            break; // Stop on first failure
        }
    }
    
    BOOST_CHECK_GT(successful, 0);
    BOOST_CHECK_LE(successful, utxoSet.GetMaxUTXOs());
    
    // Should not crash or corrupt memory
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), successful);
}

BOOST_FIXTURE_TEST_CASE(memory_script_blob_overflow, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize(10 * 1024 * 1024)); // Small limit
    
    // Try to overflow script blob
    int successful = 0;
    
    for (int i = 0; i < 100000; i++) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = 1000; // Large scripts
        header.txid_index = i;
        
        std::vector<uint8_t> script(1000, 0xFF);
        
        if (!utxoSet.AddUTXO(txid, 0, header, script.data())) {
            break; // Script blob full
        }
        successful++;
    }
    
    BOOST_CHECK_GT(successful, 0);
    BOOST_CHECK_LE(utxoSet.GetScriptBlobUsed(), utxoSet.GetScriptBlobSize());
}

BOOST_FIXTURE_TEST_CASE(memory_cleanup_on_destruction, BasicTestingSetup)
{
    size_t free_before, total_before;
    cudaMemGetInfo(&free_before, &total_before);
    
    {
        gpu::GPUUTXOSet utxoSet;
        BOOST_REQUIRE(utxoSet.Initialize());
        
        // Add some data
        for (int i = 0; i < 1000; i++) {
            uint256 txid = GetRandHash();
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.txid_index = i;
            utxoSet.AddUTXO(txid, 0, header, nullptr);
        }
        
        size_t free_during, total_during;
        cudaMemGetInfo(&free_during, &total_during);
        BOOST_CHECK_LT(free_during, free_before);
    } // utxoSet destroyed here
    
    // Memory should be freed
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    
    // Allow small difference for driver overhead
    BOOST_CHECK_CLOSE(static_cast<double>(free_after), 
                      static_cast<double>(free_before), 1.0);
}

BOOST_FIXTURE_TEST_CASE(memory_header_alignment, BasicTestingSetup)
{
    // Verify header structure is properly aligned
    BOOST_CHECK_EQUAL(sizeof(gpu::UTXOHeader), 32);
    BOOST_CHECK_EQUAL(alignof(gpu::UTXOHeader), 8);
    
    // Check field offsets
    gpu::UTXOHeader header;
    BOOST_CHECK_EQUAL(offsetof(gpu::UTXOHeader, amount), 0);
    BOOST_CHECK_EQUAL(offsetof(gpu::UTXOHeader, script_offset), 8);
    BOOST_CHECK_LE(offsetof(gpu::UTXOHeader, padding), 26);
}

BOOST_FIXTURE_TEST_CASE(memory_concurrent_allocations, BasicTestingSetup)
{
    // Test multiple UTXO sets sharing VRAM
    std::vector<std::unique_ptr<gpu::GPUUTXOSet>> sets;
    
    // Try to create multiple sets
    for (int i = 0; i < 10; i++) {
        auto set = std::make_unique<gpu::GPUUTXOSet>();
        if (set->Initialize(10 * 1024 * 1024)) { // 10MB each
            sets.push_back(std::move(set));
        } else {
            break; // Out of memory
        }
    }
    
    BOOST_CHECK_GT(sets.size(), 0);
    
    // Each should work independently
    for (size_t i = 0; i < sets.size(); i++) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        
        BOOST_CHECK(sets[i]->AddUTXO(txid, i, header, nullptr));
        BOOST_CHECK(sets[i]->HasUTXO(txid, i));
    }
}

BOOST_FIXTURE_TEST_CASE(memory_stress_allocation_deallocation, BasicTestingSetup)
{
    // Stress test allocation/deallocation cycles
    for (int cycle = 0; cycle < 10; cycle++) {
        gpu::GPUUTXOSet utxoSet;
        BOOST_CHECK(utxoSet.Initialize(50 * 1024 * 1024));
        
        // Add random data
        for (int i = 0; i < 1000; i++) {
            uint256 txid = GetRandHash();
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.script_size = GetRand<uint16_t>() % 100;
            header.txid_index = i;
            
            std::vector<uint8_t> script(header.script_size, i % 256);
            utxoSet.AddUTXO(txid, 0, header, script.data());
        }
    }
    
    // Check for memory leaks
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Should have most memory available
    BOOST_CHECK_GT(free_mem, total_mem * 0.8);
}

BOOST_AUTO_TEST_SUITE_END()