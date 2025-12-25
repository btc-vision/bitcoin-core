// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>
#include <gpu_kernel/gpu_tx_graph.h>
#include <cstring>
#include <random>

BOOST_AUTO_TEST_SUITE(gpu_tx_graph_tests)

// Helper to create a fake txid from a number
static void MakeTxid(uint8_t* txid, uint32_t value) {
    memset(txid, 0, 32);
    memcpy(txid, &value, sizeof(value));
}

// Helper to create a TxForGraph
static gpu::TxForGraph MakeTx(uint32_t id, uint32_t index,
                               const std::vector<std::pair<uint32_t, uint32_t>>& input_refs = {}) {
    gpu::TxForGraph tx;
    MakeTxid(tx.txid, id);
    tx.tx_index = index;
    tx.num_outputs = 2;  // Default 2 outputs per tx

    for (const auto& [prev_id, prev_vout] : input_refs) {
        gpu::TxInputRef input;
        MakeTxid(input.prev_txid, prev_id);
        input.prev_vout = prev_vout;
        tx.inputs.push_back(input);
    }

    return tx;
}

BOOST_AUTO_TEST_CASE(empty_graph)
{
    gpu::TxDependencyGraph graph;
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 0u);
    BOOST_CHECK_EQUAL(result.num_batches, 0u);
    BOOST_CHECK(!result.has_cycle);
}

BOOST_AUTO_TEST_CASE(single_transaction)
{
    gpu::TxDependencyGraph graph;

    auto tx = MakeTx(1, 0);
    graph.AddTransaction(tx);
    graph.BuildDependencies();

    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 1u);
    BOOST_CHECK_EQUAL(result.num_batches, 1u);
    BOOST_CHECK_EQUAL(result.batches[0].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[0][0], 0u);  // tx_index = 0
    BOOST_CHECK(!result.has_cycle);
}

BOOST_AUTO_TEST_CASE(independent_transactions)
{
    // All transactions are independent - should be in one batch
    gpu::TxDependencyGraph graph;

    for (uint32_t i = 0; i < 10; i++) {
        auto tx = MakeTx(i + 100, i);  // txid 100-109, index 0-9
        graph.AddTransaction(tx);
    }
    graph.BuildDependencies();

    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 10u);
    BOOST_CHECK_EQUAL(result.num_batches, 1u);
    BOOST_CHECK_EQUAL(result.batches[0].size(), 10u);
    BOOST_CHECK_EQUAL(result.num_with_dependencies, 0u);
    BOOST_CHECK(!result.has_cycle);
}

BOOST_AUTO_TEST_CASE(simple_chain)
{
    // TX_A -> TX_B -> TX_C (linear chain)
    gpu::TxDependencyGraph graph;

    // TX_A (no dependencies)
    auto tx_a = MakeTx(1, 0);
    graph.AddTransaction(tx_a);

    // TX_B spends TX_A output 0
    auto tx_b = MakeTx(2, 1, {{1, 0}});
    graph.AddTransaction(tx_b);

    // TX_C spends TX_B output 0
    auto tx_c = MakeTx(3, 2, {{2, 0}});
    graph.AddTransaction(tx_c);

    graph.BuildDependencies();
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 3u);
    BOOST_CHECK_EQUAL(result.num_batches, 3u);
    BOOST_CHECK_EQUAL(result.num_with_dependencies, 2u);
    BOOST_CHECK(!result.has_cycle);

    // Batch 0: TX_A (index 0)
    // Batch 1: TX_B (index 1)
    // Batch 2: TX_C (index 2)
    BOOST_CHECK_EQUAL(result.batches[0].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[0][0], 0u);

    BOOST_CHECK_EQUAL(result.batches[1].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[1][0], 1u);

    BOOST_CHECK_EQUAL(result.batches[2].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[2][0], 2u);
}

BOOST_AUTO_TEST_CASE(diamond_dependency)
{
    //     TX_A
    //    /    \
    //  TX_B  TX_C
    //    \    /
    //     TX_D

    gpu::TxDependencyGraph graph;

    // TX_A (no dependencies)
    auto tx_a = MakeTx(1, 0);
    graph.AddTransaction(tx_a);

    // TX_B spends TX_A output 0
    auto tx_b = MakeTx(2, 1, {{1, 0}});
    graph.AddTransaction(tx_b);

    // TX_C spends TX_A output 1
    auto tx_c = MakeTx(3, 2, {{1, 1}});
    graph.AddTransaction(tx_c);

    // TX_D spends TX_B output 0 and TX_C output 0
    auto tx_d = MakeTx(4, 3, {{2, 0}, {3, 0}});
    graph.AddTransaction(tx_d);

    graph.BuildDependencies();
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 4u);
    BOOST_CHECK_EQUAL(result.num_batches, 3u);
    BOOST_CHECK(!result.has_cycle);

    // Batch 0: TX_A
    BOOST_CHECK_EQUAL(result.batches[0].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[0][0], 0u);

    // Batch 1: TX_B and TX_C (can run in parallel)
    BOOST_CHECK_EQUAL(result.batches[1].size(), 2u);
    // Order within batch doesn't matter, but both should be present
    bool has_b = result.batches[1][0] == 1 || result.batches[1][1] == 1;
    bool has_c = result.batches[1][0] == 2 || result.batches[1][1] == 2;
    BOOST_CHECK(has_b && has_c);

    // Batch 2: TX_D
    BOOST_CHECK_EQUAL(result.batches[2].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[2][0], 3u);
}

BOOST_AUTO_TEST_CASE(mixed_dependencies)
{
    // Mix of independent and dependent transactions
    //
    // TX_A (independent)
    // TX_B (independent)
    // TX_C depends on TX_A
    // TX_D depends on TX_B
    // TX_E (independent)
    // TX_F depends on TX_C

    gpu::TxDependencyGraph graph;

    auto tx_a = MakeTx(1, 0);  // Independent
    auto tx_b = MakeTx(2, 1);  // Independent
    auto tx_c = MakeTx(3, 2, {{1, 0}});  // Depends on TX_A
    auto tx_d = MakeTx(4, 3, {{2, 0}});  // Depends on TX_B
    auto tx_e = MakeTx(5, 4);  // Independent
    auto tx_f = MakeTx(6, 5, {{3, 0}});  // Depends on TX_C

    graph.AddTransaction(tx_a);
    graph.AddTransaction(tx_b);
    graph.AddTransaction(tx_c);
    graph.AddTransaction(tx_d);
    graph.AddTransaction(tx_e);
    graph.AddTransaction(tx_f);

    graph.BuildDependencies();
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 6u);
    BOOST_CHECK_EQUAL(result.num_batches, 3u);
    BOOST_CHECK(!result.has_cycle);

    // Batch 0: TX_A, TX_B, TX_E (all independent)
    BOOST_CHECK_EQUAL(result.batches[0].size(), 3u);

    // Batch 1: TX_C, TX_D (depend on batch 0)
    BOOST_CHECK_EQUAL(result.batches[1].size(), 2u);

    // Batch 2: TX_F (depends on batch 1)
    BOOST_CHECK_EQUAL(result.batches[2].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[2][0], 5u);
}

BOOST_AUTO_TEST_CASE(external_dependencies_only)
{
    // All transactions depend on UTXOs from outside the block
    // (not on each other) - should all be in batch 0

    gpu::TxDependencyGraph graph;

    // TX_A spends external UTXO (txid 999)
    auto tx_a = MakeTx(1, 0, {{999, 0}});
    graph.AddTransaction(tx_a);

    // TX_B spends different external UTXO
    auto tx_b = MakeTx(2, 1, {{998, 0}});
    graph.AddTransaction(tx_b);

    // TX_C spends different external UTXO
    auto tx_c = MakeTx(3, 2, {{997, 0}});
    graph.AddTransaction(tx_c);

    graph.BuildDependencies();
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 3u);
    BOOST_CHECK_EQUAL(result.num_batches, 1u);
    BOOST_CHECK_EQUAL(result.batches[0].size(), 3u);
    BOOST_CHECK_EQUAL(result.num_with_dependencies, 0u);
    BOOST_CHECK(!result.has_cycle);
}

BOOST_AUTO_TEST_CASE(multiple_inputs_same_parent)
{
    // TX_B spends multiple outputs from TX_A

    gpu::TxDependencyGraph graph;

    auto tx_a = MakeTx(1, 0);
    tx_a.num_outputs = 3;
    graph.AddTransaction(tx_a);

    // TX_B spends outputs 0, 1, and 2 from TX_A
    auto tx_b = MakeTx(2, 1, {{1, 0}, {1, 1}, {1, 2}});
    graph.AddTransaction(tx_b);

    graph.BuildDependencies();
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 2u);
    BOOST_CHECK_EQUAL(result.num_batches, 2u);
    BOOST_CHECK_EQUAL(result.num_with_dependencies, 1u);
    BOOST_CHECK(!result.has_cycle);
}

BOOST_AUTO_TEST_CASE(wide_parallel_tree)
{
    // TX_A creates 100 outputs
    // TX_B1-TX_B100 each spend one output from TX_A

    gpu::TxDependencyGraph graph;

    auto tx_a = MakeTx(1, 0);
    tx_a.num_outputs = 100;
    graph.AddTransaction(tx_a);

    for (uint32_t i = 0; i < 100; i++) {
        auto tx_child = MakeTx(100 + i, i + 1, {{1, i}});
        graph.AddTransaction(tx_child);
    }

    graph.BuildDependencies();
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 101u);
    BOOST_CHECK_EQUAL(result.num_batches, 2u);
    BOOST_CHECK(!result.has_cycle);

    // Batch 0: TX_A only
    BOOST_CHECK_EQUAL(result.batches[0].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[0][0], 0u);

    // Batch 1: All 100 children (can run in parallel)
    BOOST_CHECK_EQUAL(result.batches[1].size(), 100u);
    BOOST_CHECK_EQUAL(result.max_batch_size, 100u);
}

BOOST_AUTO_TEST_CASE(utility_function_test)
{
    // Test the SortTransactionsForGPU utility function
    std::vector<gpu::TxForGraph> transactions;

    transactions.push_back(MakeTx(1, 0));  // Independent
    transactions.push_back(MakeTx(2, 1, {{1, 0}}));  // Depends on tx 1
    transactions.push_back(MakeTx(3, 2));  // Independent

    auto result = gpu::SortTransactionsForGPU(transactions);

    BOOST_CHECK_EQUAL(result.total_txs, 3u);
    BOOST_CHECK_EQUAL(result.num_batches, 2u);
    BOOST_CHECK(!result.has_cycle);

    // Batch 0: tx indices 0 and 2 (independent)
    BOOST_CHECK_EQUAL(result.batches[0].size(), 2u);

    // Batch 1: tx index 1 (depends on tx 0)
    BOOST_CHECK_EQUAL(result.batches[1].size(), 1u);
    BOOST_CHECK_EQUAL(result.batches[1][0], 1u);
}

BOOST_AUTO_TEST_CASE(realistic_block_pattern)
{
    // Simulate a realistic block pattern:
    // - Coinbase (tx 0) - independent
    // - Several independent regular transactions
    // - Some transactions spending from others in the block

    gpu::TxDependencyGraph graph;

    // Coinbase
    auto coinbase = MakeTx(0, 0);
    coinbase.num_outputs = 1;
    graph.AddTransaction(coinbase);

    // 5 independent transactions spending external UTXOs
    for (uint32_t i = 1; i <= 5; i++) {
        auto tx = MakeTx(i, i, {{1000 + i, 0}});  // External UTXO
        tx.num_outputs = 2;
        graph.AddTransaction(tx);
    }

    // 2 transactions spending coinbase reward
    auto spend_cb1 = MakeTx(6, 6, {{0, 0}});  // Spends coinbase
    graph.AddTransaction(spend_cb1);

    // 1 transaction spending from tx 1
    auto spend_tx1 = MakeTx(7, 7, {{1, 0}});
    graph.AddTransaction(spend_tx1);

    // 1 transaction spending from both tx 2 and tx 3
    auto spend_tx23 = MakeTx(8, 8, {{2, 0}, {3, 1}});
    graph.AddTransaction(spend_tx23);

    graph.BuildDependencies();
    auto result = graph.ComputeBatches();

    BOOST_CHECK_EQUAL(result.total_txs, 9u);
    BOOST_CHECK(!result.has_cycle);

    // Should have 2 batches:
    // Batch 0: coinbase + independent txs (0,1,2,3,4,5) = 6 txs
    // Batch 1: spend_cb1 + spend_tx1 + spend_tx23 (6,7,8) = 3 txs
    BOOST_CHECK_EQUAL(result.num_batches, 2u);
    BOOST_CHECK_EQUAL(result.batches[0].size(), 6u);
    BOOST_CHECK_EQUAL(result.batches[1].size(), 3u);
}

BOOST_AUTO_TEST_SUITE_END()
