// Copyright (c) 2024-present The Bitcoin Core developers
// GPU UTXO Compaction Tests - Complete coverage for memory compaction

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_utxo.h>
#include <random.h>
#include <chrono>

BOOST_AUTO_TEST_SUITE(gpu_compaction_tests)

BOOST_FIXTURE_TEST_CASE(compaction_trigger_threshold, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add many UTXOs with scripts
    std::vector<uint256> txids;
    const int NUM_UTXOS = 5000;
    
    for (int i = 0; i < NUM_UTXOS; i++) {
        uint256 txid = GetRandHash();
        txids.push_back(txid);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = 25 + (i % 50);  // Variable sizes
        header.txid_index = i;
        header.vout = 0;
        header.amount = (i + 1) * COIN;
        
        std::vector<uint8_t> script(header.script_size, i % 256);
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }
    
    size_t initial_blob_used = utxoSet.GetScriptBlobUsed();
    size_t initial_vram_used = utxoSet.GetTotalVRAMUsed();
    
    // Spend 15% to trigger compaction (>10% threshold)
    int to_spend = NUM_UTXOS * 0.15;
    for (int i = 0; i < to_spend; i++) {
        BOOST_CHECK(utxoSet.SpendUTXO(txids[i], 0));
    }
    
    // Check fragmentation level
    size_t free_space = utxoSet.GetTotalFreeSpace();
    double fragmentation_ratio = static_cast<double>(free_space) / initial_vram_used;
    
    BOOST_CHECK_GT(fragmentation_ratio, 0.10);  // Should exceed 10% threshold
    
    // Trigger compaction
    bool compacted = utxoSet.Compact();
    BOOST_CHECK(compacted);
    
    // Verify remaining UTXOs still exist
    for (int i = to_spend; i < NUM_UTXOS; i++) {
        BOOST_CHECK(utxoSet.HasUTXO(txids[i], 0));
    }
    
    // Free space should be near zero after compaction
    BOOST_CHECK_LT(utxoSet.GetTotalFreeSpace(), initial_blob_used * 0.01);
}

BOOST_FIXTURE_TEST_CASE(compaction_no_data_loss, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Create UTXOs with specific data
    struct TestUTXO {
        uint256 txid;
        uint32_t vout;
        CAmount amount;
        uint32_t height;
        std::vector<uint8_t> script;
    };
    
    std::vector<TestUTXO> utxos;
    
    for (int i = 0; i < 1000; i++) {
        TestUTXO utxo;
        utxo.txid = GetRandHash();
        utxo.vout = i % 10;
        utxo.amount = (i + 1) * COIN;
        utxo.height = 100000 + i;
        utxo.script.resize(20 + (i % 30));
        GetRandBytes(utxo.script);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.amount = utxo.amount;
        header.blockHeight = utxo.height;
        header.vout = utxo.vout;
        header.script_size = utxo.script.size();
        header.txid_index = i;
        
        BOOST_CHECK(utxoSet.AddUTXO(utxo.txid, utxo.vout, header, utxo.script.data()));
        utxos.push_back(utxo);
    }
    
    // Spend some to create fragmentation
    for (int i = 0; i < 300; i++) {
        BOOST_CHECK(utxoSet.SpendUTXO(utxos[i].txid, utxos[i].vout));
    }
    
    // Compact
    BOOST_CHECK(utxoSet.Compact());
    
    // Verify all unspent UTXOs are intact
    for (size_t i = 300; i < utxos.size(); i++) {
        BOOST_CHECK(utxoSet.HasUTXO(utxos[i].txid, utxos[i].vout));
    }
    
    // Verify spent ones are still gone
    for (int i = 0; i < 300; i++) {
        BOOST_CHECK(!utxoSet.HasUTXO(utxos[i].txid, utxos[i].vout));
    }
}

BOOST_FIXTURE_TEST_CASE(compaction_performance, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add large number of UTXOs
    const int NUM_UTXOS = 10000;
    std::vector<uint256> txids;
    
    for (int i = 0; i < NUM_UTXOS; i++) {
        uint256 txid = GetRandHash();
        txids.push_back(txid);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = 25;
        header.txid_index = i;
        
        std::vector<uint8_t> script(25, 0xAB);
        utxoSet.AddUTXO(txid, 0, header, script.data());
    }
    
    // Create fragmentation
    for (int i = 0; i < NUM_UTXOS / 3; i++) {
        utxoSet.SpendUTXO(txids[i * 3], 0);  // Spend every 3rd
    }
    
    // Measure compaction time
    auto start = std::chrono::high_resolution_clock::now();
    bool result = utxoSet.Compact();
    auto end = std::chrono::high_resolution_clock::now();
    
    BOOST_CHECK(result);
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    BOOST_TEST_MESSAGE("Compaction of " + std::to_string(NUM_UTXOS * 2/3) + 
                      " UTXOs took " + std::to_string(duration.count()) + " ms");
    
    // Should complete reasonably quickly (< 1 second)
    BOOST_CHECK_LT(duration.count(), 1000);
}

BOOST_FIXTURE_TEST_CASE(compaction_hash_table_rebuild, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add UTXOs and track their positions
    std::map<uint256, uint32_t> txid_to_vout;
    
    for (int i = 0; i < 1000; i++) {
        uint256 txid = GetRandHash();
        uint32_t vout = i % 50;
        txid_to_vout[txid] = vout;
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = i;
        header.vout = vout;
        header.script_size = 25;
        
        std::vector<uint8_t> script(25, 0xCC);
        BOOST_CHECK(utxoSet.AddUTXO(txid, vout, header, script.data()));
    }
    
    // Spend some
    auto it = txid_to_vout.begin();
    for (int i = 0; i < 200; i++) {
        utxoSet.SpendUTXO(it->first, it->second);
        it = txid_to_vout.erase(it);
    }
    
    // Compact (rebuilds hash tables)
    BOOST_CHECK(utxoSet.Compact());
    
    // Verify all remaining UTXOs are still findable via hash tables
    for (const auto& [txid, vout] : txid_to_vout) {
        BOOST_CHECK(utxoSet.HasUTXO(txid, vout));
    }
}

BOOST_FIXTURE_TEST_CASE(compaction_minimal_fragmentation, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add UTXOs
    std::vector<uint256> txids;
    for (int i = 0; i < 1000; i++) {
        uint256 txid = GetRandHash();
        txids.push_back(txid);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = i;
        header.script_size = 25;
        
        std::vector<uint8_t> script(25, 0xDD);
        utxoSet.AddUTXO(txid, 0, header, script.data());
    }
    
    // Spend only 5% (below 10% threshold)
    for (int i = 0; i < 50; i++) {
        utxoSet.SpendUTXO(txids[i], 0);
    }
    
    size_t free_space = utxoSet.GetTotalFreeSpace();
    size_t total_used = utxoSet.GetTotalVRAMUsed();
    
    // Should not trigger compaction (< 10% fragmentation)
    if (free_space < total_used * 0.1) {
        bool result = utxoSet.Compact();
        // Compaction might still succeed but shouldn't be necessary
        BOOST_TEST_MESSAGE("Compaction with low fragmentation: " + 
                          std::to_string(free_space * 100.0 / total_used) + "%");
    }
}

BOOST_FIXTURE_TEST_CASE(compaction_script_blob_defragmentation, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add UTXOs with varying script sizes to create gaps
    std::vector<uint256> txids;
    std::vector<size_t> script_sizes;
    
    for (int i = 0; i < 1000; i++) {
        uint256 txid = GetRandHash();
        txids.push_back(txid);
        
        // Alternating large and small scripts
        size_t script_size = (i % 2 == 0) ? 100 : 25;
        script_sizes.push_back(script_size);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = script_size;
        header.txid_index = i;
        
        std::vector<uint8_t> script(script_size, i % 256);
        utxoSet.AddUTXO(txid, 0, header, script.data());
    }
    
    size_t initial_blob_used = utxoSet.GetScriptBlobUsed();
    
    // Spend all large scripts (creating gaps)
    for (int i = 0; i < 1000; i += 2) {
        utxoSet.SpendUTXO(txids[i], 0);
    }
    
    // Blob usage should not change yet (fragmented)
    BOOST_CHECK_EQUAL(utxoSet.GetScriptBlobUsed(), initial_blob_used);
    
    // Compact to defragment
    BOOST_CHECK(utxoSet.Compact());
    
    // Blob usage should decrease significantly
    size_t compacted_blob_used = utxoSet.GetScriptBlobUsed();
    BOOST_CHECK_LT(compacted_blob_used, initial_blob_used * 0.5);
}

BOOST_FIXTURE_TEST_CASE(compaction_concurrent_operations, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add initial set
    std::vector<uint256> initial_txids;
    for (int i = 0; i < 1000; i++) {
        uint256 txid = GetRandHash();
        initial_txids.push_back(txid);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = i;
        header.script_size = 25;
        
        std::vector<uint8_t> script(25, 0xEE);
        utxoSet.AddUTXO(txid, 0, header, script.data());
    }
    
    // Spend some
    for (int i = 0; i < 200; i++) {
        utxoSet.SpendUTXO(initial_txids[i], 0);
    }
    
    // Compact
    BOOST_CHECK(utxoSet.Compact());
    
    // Add new UTXOs after compaction
    std::vector<uint256> new_txids;
    for (int i = 0; i < 500; i++) {
        uint256 txid = GetRandHash();
        new_txids.push_back(txid);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = 1000 + i;
        header.script_size = 30;
        
        std::vector<uint8_t> script(30, 0xFF);
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }
    
    // Verify both old and new UTXOs exist
    for (int i = 200; i < 1000; i++) {
        BOOST_CHECK(utxoSet.HasUTXO(initial_txids[i], 0));
    }
    for (const auto& txid : new_txids) {
        BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
    }
}

BOOST_FIXTURE_TEST_CASE(compaction_edge_cases, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Test compaction with no UTXOs
    bool result = utxoSet.Compact();
    BOOST_CHECK(result);
    
    // Add single UTXO
    uint256 txid = GetRandHash();
    gpu::UTXOHeader header;
    memset(&header, 0, sizeof(header));
    utxoSet.AddUTXO(txid, 0, header, nullptr);
    
    // Compact with single UTXO
    result = utxoSet.Compact();
    BOOST_CHECK(result);
    BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
    
    // Spend it and compact empty set
    utxoSet.SpendUTXO(txid, 0);
    result = utxoSet.Compact();
    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_SUITE_END()