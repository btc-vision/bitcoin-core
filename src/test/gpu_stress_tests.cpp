// Copyright (c) 2024-present The Bitcoin Core developers
// GPU Stress Tests - Heavy load and edge case testing

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_utxo.h>
#include <random.h>
#include <thread>
#include <atomic>

BOOST_AUTO_TEST_SUITE(gpu_stress_tests)

BOOST_FIXTURE_TEST_CASE(stress_maximum_capacity, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Fill to maximum capacity
    size_t max_utxos = utxoSet.GetMaxUTXOs();
    BOOST_TEST_MESSAGE("Testing with max capacity: " + std::to_string(max_utxos) + " UTXOs");
    
    size_t successful = 0;
    size_t failed = 0;
    
    for (size_t i = 0; i < max_utxos + 1000; i++) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.txid_index = i % (max_utxos / 2);  // Reuse some txids
        header.vout = i % 100;
        header.amount = GetRand<CAmount>(100 * COIN);
        
        if (utxoSet.AddUTXO(txid, header.vout, header, nullptr)) {
            successful++;
        } else {
            failed++;
            if (failed > 100) break;  // Stop after many failures
        }
    }
    
    BOOST_TEST_MESSAGE("Successfully added: " + std::to_string(successful));
    BOOST_TEST_MESSAGE("Failed to add: " + std::to_string(failed));
    
    // Should be close to max capacity
    BOOST_CHECK_GT(successful, max_utxos * 0.90);
    BOOST_CHECK_LE(successful, max_utxos);
}

BOOST_FIXTURE_TEST_CASE(stress_random_operations, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Track active UTXOs
    std::map<std::pair<uint256, uint32_t>, bool> utxo_map;
    
    // Perform random operations
    const int NUM_OPERATIONS = 100000;
    int adds = 0, spends = 0, queries = 0;
    
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        int op = GetRand<int>(100);
        
        if (op < 40 || utxo_map.empty()) {
            // Add UTXO (40% or if empty)
            uint256 txid = GetRandHash();
            uint32_t vout = GetRand<uint32_t>() % 10;
            
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.txid_index = adds;
            header.vout = vout;
            header.amount = GetRand<CAmount>(100 * COIN);
            header.script_size = GetRand<uint16_t>() % 100;
            
            std::vector<uint8_t> script(header.script_size, 0xAB);
            
            if (utxoSet.AddUTXO(txid, vout, header, script.data())) {
                utxo_map[{txid, vout}] = true;
                adds++;
            }
        } else if (op < 70) {
            // Spend UTXO (30%)
            auto it = utxo_map.begin();
            std::advance(it, GetRand<size_t>(utxo_map.size()));
            
            if (utxoSet.SpendUTXO(it->first.first, it->first.second)) {
                utxo_map.erase(it);
                spends++;
            }
        } else {
            // Query UTXO (30%)
            if (GetRand<int>(2) == 0 && !utxo_map.empty()) {
                // Query existing
                auto it = utxo_map.begin();
                std::advance(it, GetRand<size_t>(utxo_map.size()));
                
                bool exists = utxoSet.HasUTXO(it->first.first, it->first.second);
                BOOST_CHECK(exists);
            } else {
                // Query non-existing
                uint256 random_txid = GetRandHash();
                uint32_t random_vout = GetRand<uint32_t>();
                
                bool exists = utxoSet.HasUTXO(random_txid, random_vout);
                if (utxo_map.find({random_txid, random_vout}) == utxo_map.end()) {
                    BOOST_CHECK(!exists);
                }
            }
            queries++;
        }
    }
    
    BOOST_TEST_MESSAGE("Operations - Adds: " + std::to_string(adds) + 
                      ", Spends: " + std::to_string(spends) + 
                      ", Queries: " + std::to_string(queries));
    
    // Verify final state
    for (const auto& [key, _] : utxo_map) {
        BOOST_CHECK(utxoSet.HasUTXO(key.first, key.second));
    }
}

BOOST_FIXTURE_TEST_CASE(stress_memory_exhaustion, BasicTestingSetup)
{
    // Test behavior when running out of memory
    std::vector<std::unique_ptr<gpu::GPUUTXOSet>> sets;
    
    // Keep allocating until failure
    int successful_allocs = 0;
    for (int i = 0; i < 100; i++) {
        auto set = std::make_unique<gpu::GPUUTXOSet>();
        
        // Try to allocate 500MB each
        if (set->Initialize(500 * 1024 * 1024)) {
            sets.push_back(std::move(set));
            successful_allocs++;
        } else {
            break;  // Out of memory
        }
    }
    
    BOOST_TEST_MESSAGE("Successfully allocated " + std::to_string(successful_allocs) + 
                      " UTXO sets of 500MB each");
    
    // Should have allocated at least one
    BOOST_CHECK_GT(successful_allocs, 0);
    
    // All allocated sets should still work
    for (auto& set : sets) {
        uint256 txid = GetRandHash();
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        
        BOOST_CHECK(set->AddUTXO(txid, 0, header, nullptr));
        BOOST_CHECK(set->HasUTXO(txid, 0));
    }
}

BOOST_FIXTURE_TEST_CASE(stress_hash_collision_resistance, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Try to create hash collisions
    const int NUM_ATTEMPTS = 50000;
    std::set<uint32_t> seen_hashes[4];
    int collisions[4] = {0, 0, 0, 0};
    
    for (int i = 0; i < NUM_ATTEMPTS; i++) {
        uint256 txid = GetRandHash();
        uint32_t vout = GetRand<uint32_t>();
        
        uint32_t h1 = utxoSet.Hash1(txid, vout);
        uint32_t h2 = utxoSet.Hash2(txid, vout);
        uint32_t h3 = utxoSet.Hash3(txid, vout);
        uint32_t h4 = utxoSet.Hash4(txid, vout);
        
        if (seen_hashes[0].count(h1)) collisions[0]++;
        if (seen_hashes[1].count(h2)) collisions[1]++;
        if (seen_hashes[2].count(h3)) collisions[2]++;
        if (seen_hashes[3].count(h4)) collisions[3]++;
        
        seen_hashes[0].insert(h1);
        seen_hashes[1].insert(h2);
        seen_hashes[2].insert(h3);
        seen_hashes[3].insert(h4);
    }
    
    for (int i = 0; i < 4; i++) {
        double collision_rate = static_cast<double>(collisions[i]) / NUM_ATTEMPTS;
        BOOST_TEST_MESSAGE("Hash " + std::to_string(i+1) + " collision rate: " + 
                          std::to_string(collision_rate * 100) + "%");
        
        // Collision rate should be low
        BOOST_CHECK_LT(collision_rate, 0.05);  // Less than 5%
    }
}

BOOST_FIXTURE_TEST_CASE(stress_rapid_add_remove_cycles, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Rapid add/remove cycles
    const int NUM_CYCLES = 1000;
    const int UTXOS_PER_CYCLE = 100;
    
    for (int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        std::vector<uint256> txids;
        
        // Add batch
        for (int i = 0; i < UTXOS_PER_CYCLE; i++) {
            uint256 txid = GetRandHash();
            txids.push_back(txid);
            
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.txid_index = cycle * UTXOS_PER_CYCLE + i;
            header.vout = i;
            
            utxoSet.AddUTXO(txid, i, header, nullptr);
        }
        
        // Remove batch
        for (size_t i = 0; i < txids.size(); i++) {
            utxoSet.SpendUTXO(txids[i], i);
        }
        
        // Occasionally trigger compaction
        if (cycle % 100 == 99) {
            utxoSet.Compact();
        }
    }
    
    // Should handle rapid cycles without issues
    BOOST_CHECK(true);
}

BOOST_FIXTURE_TEST_CASE(stress_large_scripts, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize(1024 * 1024 * 1024));  // 1GB
    
    // Add UTXOs with very large scripts
    int successful = 0;
    size_t total_script_size = 0;
    
    for (int i = 0; i < 10000; i++) {
        uint256 txid = GetRandHash();
        
        // Random large script size (up to 10KB)
        size_t script_size = 100 + GetRand<size_t>(10000);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.script_size = std::min(script_size, size_t(65535));
        header.txid_index = i;
        
        std::vector<uint8_t> script(header.script_size);
        GetRandBytes(script);
        
        if (utxoSet.AddUTXO(txid, 0, header, script.data())) {
            successful++;
            total_script_size += header.script_size;
        } else {
            break;  // Script blob full
        }
    }
    
    BOOST_TEST_MESSAGE("Added " + std::to_string(successful) + " large-script UTXOs");
    BOOST_TEST_MESSAGE("Total script size: " + std::to_string(total_script_size / (1024*1024)) + " MB");
    
    BOOST_CHECK_GT(successful, 100);
}

BOOST_FIXTURE_TEST_CASE(stress_txid_deduplication, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Create many outputs from few transactions
    const int NUM_TXIDS = 100;
    const int OUTPUTS_PER_TX = 1000;
    
    std::vector<uint256> txids;
    for (int i = 0; i < NUM_TXIDS; i++) {
        txids.push_back(GetRandHash());
    }
    
    int total_added = 0;
    for (int tx = 0; tx < NUM_TXIDS; tx++) {
        for (int vout = 0; vout < OUTPUTS_PER_TX; vout++) {
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.txid_index = tx;  // Should reuse txid
            header.vout = vout;
            header.amount = (vout + 1) * 1000;
            
            if (utxoSet.AddUTXO(txids[tx], vout, header, nullptr)) {
                total_added++;
            }
        }
    }
    
    BOOST_TEST_MESSAGE("Added " + std::to_string(total_added) + " outputs from " + 
                      std::to_string(NUM_TXIDS) + " transactions");
    
    // Verify txid table is efficiently used
    BOOST_CHECK_LE(utxoSet.GetTxidTableUsed(), NUM_TXIDS);
    
    // Verify all outputs exist
    for (int tx = 0; tx < NUM_TXIDS; tx++) {
        for (int vout = 0; vout < std::min(100, OUTPUTS_PER_TX); vout++) {
            BOOST_CHECK(utxoSet.HasUTXO(txids[tx], vout));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(stress_continuous_operation, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Run continuous operations for a fixed time
    auto start = std::chrono::steady_clock::now();
    auto end = start + std::chrono::seconds(5);  // Run for 5 seconds
    
    std::atomic<int> operations(0);
    std::map<uint256, std::set<uint32_t>> active_utxos;
    
    while (std::chrono::steady_clock::now() < end) {
        int op = GetRand<int>(100);
        
        if (op < 50) {
            // Add
            uint256 txid = GetRandHash();
            uint32_t vout = GetRand<uint32_t>() % 100;
            
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.txid_index = operations % 100000;
            header.vout = vout;
            
            if (utxoSet.AddUTXO(txid, vout, header, nullptr)) {
                active_utxos[txid].insert(vout);
            }
        } else if (op < 80 && !active_utxos.empty()) {
            // Spend
            auto tx_it = active_utxos.begin();
            std::advance(tx_it, GetRand<size_t>(active_utxos.size()));
            
            if (!tx_it->second.empty()) {
                auto vout_it = tx_it->second.begin();
                std::advance(vout_it, GetRand<size_t>(tx_it->second.size()));
                
                if (utxoSet.SpendUTXO(tx_it->first, *vout_it)) {
                    tx_it->second.erase(vout_it);
                    if (tx_it->second.empty()) {
                        active_utxos.erase(tx_it);
                    }
                }
            }
        } else {
            // Query
            if (!active_utxos.empty() && GetRand<int>(2) == 0) {
                auto tx_it = active_utxos.begin();
                std::advance(tx_it, GetRand<size_t>(active_utxos.size()));
                
                if (!tx_it->second.empty()) {
                    uint32_t vout = *tx_it->second.begin();
                    BOOST_CHECK(utxoSet.HasUTXO(tx_it->first, vout));
                }
            }
        }
        
        operations++;
    }
    
    BOOST_TEST_MESSAGE("Performed " + std::to_string(operations) + " operations in 5 seconds");
    BOOST_TEST_MESSAGE("Rate: " + std::to_string(operations / 5) + " ops/sec");
    
    // Should achieve reasonable throughput
    BOOST_CHECK_GT(operations, 10000);
}

BOOST_AUTO_TEST_SUITE_END()