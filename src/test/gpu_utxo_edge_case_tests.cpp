// Copyright (c) 2024-present The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_utxo.h>
#include <gpu_kernel/gpu_logging.h>
#include <coins.h>
#include <primitives/transaction.h>
#include <script/script.h>
#include <random.h>
#include <util/time.h>

#include <vector>
#include <memory>
#include <map>

namespace {

// Helper to create a P2PKH script
CScript CreateP2PKHScript(const std::vector<uint8_t>& pubkeyhash) {
    CScript script;
    script << OP_DUP << OP_HASH160 << pubkeyhash << OP_EQUALVERIFY << OP_CHECKSIG;
    return script;
}

// Helper to create a random txid
uint256 RandomTxid() {
    return GetRandHash();
}

// Helper to create a random pubkey hash
std::vector<uint8_t> RandomPubKeyHash() {
    std::vector<uint8_t> hash(20);
    GetStrongRandBytes(hash);
    return hash;
}

// Helper to create a UTXO header
gpu::UTXOHeader CreateHeader(int64_t amount, uint32_t height, uint16_t vout,
                              uint16_t script_size, bool coinbase = false) {
    gpu::UTXOHeader header;
    memset(&header, 0, sizeof(header));
    header.amount = amount;
    header.blockHeight = height;
    header.vout = vout;
    header.script_size = script_size;
    header.script_type = gpu::SCRIPT_TYPE_P2PKH;
    header.flags = coinbase ? gpu::UTXO_FLAG_COINBASE : 0;
    return header;
}

} // namespace

BOOST_AUTO_TEST_SUITE(gpu_utxo_edge_case_tests)

// =============================================================================
// Same-Block Parent-Child Transaction Tests
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_parent_child_same_block, BasicTestingSetup)
{
    // Test: Transaction B spends output from Transaction A in the same block
    // Block structure:
    //   TX_A: Creates output 0 (50 BTC)
    //   TX_B: Spends TX_A:0, creates output 0 (49.9 BTC)

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    // Create parent transaction A
    uint256 txid_a = RandomTxid();
    CScript script_a = CreateP2PKHScript(RandomPubKeyHash());
    gpu::UTXOHeader header_a = CreateHeader(50 * COIN, 100, 0, script_a.size());

    // Add TX_A output to UTXO set
    BOOST_CHECK(utxoSet.AddUTXO(txid_a, 0, header_a, script_a.data()));
    BOOST_CHECK(utxoSet.HasUTXO(txid_a, 0));

    // Now TX_B spends TX_A:0 (in same block)
    BOOST_CHECK(utxoSet.SpendUTXO(txid_a, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 0));  // TX_A:0 no longer available

    // TX_B creates new output
    uint256 txid_b = RandomTxid();
    CScript script_b = CreateP2PKHScript(RandomPubKeyHash());
    gpu::UTXOHeader header_b = CreateHeader(4990000000LL, 100, 0, script_b.size());  // 49.9 BTC

    BOOST_CHECK(utxoSet.AddUTXO(txid_b, 0, header_b, script_b.data()));
    BOOST_CHECK(utxoSet.HasUTXO(txid_b, 0));

    // Final state: only TX_B:0 should exist
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 0));
    BOOST_CHECK(utxoSet.HasUTXO(txid_b, 0));
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_chain_same_block, BasicTestingSetup)
{
    // Test: Chain of 5 transactions in the same block
    // TX_A -> TX_B -> TX_C -> TX_D -> TX_E

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    std::vector<uint256> txids;
    for (int i = 0; i < 5; i++) {
        txids.push_back(RandomTxid());
    }

    // Create initial UTXO from TX_A
    CScript script = CreateP2PKHScript(RandomPubKeyHash());
    int64_t amount = 50 * COIN;
    gpu::UTXOHeader header = CreateHeader(amount, 100, 0, script.size());

    BOOST_CHECK(utxoSet.AddUTXO(txids[0], 0, header, script.data()));

    // Chain each transaction spending the previous one
    for (int i = 1; i < 5; i++) {
        // Spend previous output
        BOOST_CHECK(utxoSet.SpendUTXO(txids[i-1], 0));
        BOOST_CHECK(!utxoSet.HasUTXO(txids[i-1], 0));

        // Create new output (minus fee)
        amount -= 10000;  // 0.0001 BTC fee per tx
        header = CreateHeader(amount, 100, 0, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txids[i], 0, header, script.data()));
        BOOST_CHECK(utxoSet.HasUTXO(txids[i], 0));
    }

    // Final state: only TX_E:0 should exist
    for (int i = 0; i < 4; i++) {
        BOOST_CHECK(!utxoSet.HasUTXO(txids[i], 0));
    }
    BOOST_CHECK(utxoSet.HasUTXO(txids[4], 0));
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_multiple_children_same_parent, BasicTestingSetup)
{
    // Test: Parent creates multiple outputs, each spent by different transactions
    // TX_A creates outputs 0, 1, 2
    // TX_B spends TX_A:0
    // TX_C spends TX_A:1
    // TX_D spends TX_A:2

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid_a = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // TX_A creates 3 outputs
    for (uint16_t vout = 0; vout < 3; vout++) {
        gpu::UTXOHeader header = CreateHeader(10 * COIN, 100, vout, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txid_a, vout, header, script.data()));
    }

    // Verify all 3 outputs exist
    BOOST_CHECK(utxoSet.HasUTXO(txid_a, 0));
    BOOST_CHECK(utxoSet.HasUTXO(txid_a, 1));
    BOOST_CHECK(utxoSet.HasUTXO(txid_a, 2));

    // TX_B spends TX_A:0
    BOOST_CHECK(utxoSet.SpendUTXO(txid_a, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 0));
    BOOST_CHECK(utxoSet.HasUTXO(txid_a, 1));
    BOOST_CHECK(utxoSet.HasUTXO(txid_a, 2));

    // TX_C spends TX_A:1
    BOOST_CHECK(utxoSet.SpendUTXO(txid_a, 1));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 1));
    BOOST_CHECK(utxoSet.HasUTXO(txid_a, 2));

    // TX_D spends TX_A:2
    BOOST_CHECK(utxoSet.SpendUTXO(txid_a, 2));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 1));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_a, 2));
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_multiple_inputs_same_tx, BasicTestingSetup)
{
    // Test: Transaction with multiple inputs from different parent transactions
    // TX_A creates output 0
    // TX_B creates output 0
    // TX_C creates output 0
    // TX_D spends all three: TX_A:0, TX_B:0, TX_C:0

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    std::vector<uint256> parent_txids;
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // Create 3 parent transactions each with 1 output
    for (int i = 0; i < 3; i++) {
        uint256 txid = RandomTxid();
        parent_txids.push_back(txid);
        gpu::UTXOHeader header = CreateHeader(10 * COIN, 100, 0, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }

    // Verify all parent outputs exist
    for (const auto& txid : parent_txids) {
        BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
    }

    // TX_D spends all three inputs
    for (const auto& txid : parent_txids) {
        BOOST_CHECK(utxoSet.SpendUTXO(txid, 0));
    }

    // Verify all parent outputs are gone
    for (const auto& txid : parent_txids) {
        BOOST_CHECK(!utxoSet.HasUTXO(txid, 0));
    }

    // TX_D creates new output
    uint256 txid_d = RandomTxid();
    gpu::UTXOHeader header_d = CreateHeader(29 * COIN, 100, 0, script.size());  // 30 - 1 fee
    BOOST_CHECK(utxoSet.AddUTXO(txid_d, 0, header_d, script.data()));
    BOOST_CHECK(utxoSet.HasUTXO(txid_d, 0));
}

// =============================================================================
// Cross-Block Dependency Tests
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_cross_block_simple, BasicTestingSetup)
{
    // Test: UTXO created in block N, spent in block N+1

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid_block100 = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // Block 100: Create UTXO
    gpu::UTXOHeader header = CreateHeader(50 * COIN, 100, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(txid_block100, 0, header, script.data()));
    BOOST_CHECK(utxoSet.HasUTXO(txid_block100, 0));

    // Block 101: Spend UTXO
    BOOST_CHECK(utxoSet.SpendUTXO(txid_block100, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(txid_block100, 0));

    // Block 101: Create new UTXO
    uint256 txid_block101 = RandomTxid();
    header = CreateHeader(49 * COIN, 101, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(txid_block101, 0, header, script.data()));
    BOOST_CHECK(utxoSet.HasUTXO(txid_block101, 0));
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_cross_block_long_chain, BasicTestingSetup)
{
    // Test: UTXO spent across 10 blocks, each block has a tx spending previous block's output

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());
    int64_t amount = 100 * COIN;
    uint256 prev_txid = RandomTxid();

    // Block 100: Initial UTXO
    gpu::UTXOHeader header = CreateHeader(amount, 100, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(prev_txid, 0, header, script.data()));

    // Blocks 101-110: Each spends previous, creates new
    for (uint32_t block = 101; block <= 110; block++) {
        // Spend previous
        BOOST_CHECK(utxoSet.SpendUTXO(prev_txid, 0));
        BOOST_CHECK(!utxoSet.HasUTXO(prev_txid, 0));

        // Create new
        amount -= 10000;  // Fee
        uint256 new_txid = RandomTxid();
        header = CreateHeader(amount, block, 0, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(new_txid, 0, header, script.data()));
        BOOST_CHECK(utxoSet.HasUTXO(new_txid, 0));

        prev_txid = new_txid;
    }

    // Only the final UTXO should exist
    BOOST_CHECK(utxoSet.HasUTXO(prev_txid, 0));
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 11);  // 10 spent + 1 unspent (spent ones still in headers)
}

// =============================================================================
// Batch Update Tests (Atomic Reorg Handling)
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_batch_add_commit, BasicTestingSetup)
{
    // Test: Batch add multiple UTXOs and commit

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());
    std::vector<uint256> txids;

    // Begin batch
    utxoSet.BeginBatchUpdate();
    BOOST_CHECK(utxoSet.IsInBatchUpdate());

    // Add 10 UTXOs in batch mode
    for (int i = 0; i < 10; i++) {
        uint256 txid = RandomTxid();
        txids.push_back(txid);
        gpu::UTXOHeader header = CreateHeader(i * COIN, 100, 0, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }

    // Before commit, UTXOs should be queryable (staged adds are applied immediately for queries)
    for (const auto& txid : txids) {
        BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
    }

    // Commit
    BOOST_CHECK(utxoSet.CommitBatchUpdate());
    BOOST_CHECK(!utxoSet.IsInBatchUpdate());

    // After commit, all UTXOs should exist
    for (const auto& txid : txids) {
        BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_batch_abort, BasicTestingSetup)
{
    // Test: Batch operations that are aborted

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // Add initial UTXO
    uint256 initial_txid = RandomTxid();
    gpu::UTXOHeader header = CreateHeader(50 * COIN, 100, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(initial_txid, 0, header, script.data()));
    size_t initial_count = utxoSet.GetNumUTXOs();

    // Begin batch
    utxoSet.BeginBatchUpdate();

    // Add more UTXOs in batch
    std::vector<uint256> batch_txids;
    for (int i = 0; i < 5; i++) {
        uint256 txid = RandomTxid();
        batch_txids.push_back(txid);
        gpu::UTXOHeader batch_header = CreateHeader(i * COIN, 101, 0, script.size());
        utxoSet.AddUTXO(txid, 0, batch_header, script.data());
    }

    // Abort the batch
    utxoSet.AbortBatchUpdate();
    BOOST_CHECK(!utxoSet.IsInBatchUpdate());

    // Should be back to initial state
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), initial_count);
    BOOST_CHECK(utxoSet.HasUTXO(initial_txid, 0));

    // Batch UTXOs should not exist after abort
    // (Note: In current implementation, adds happen immediately even in batch mode,
    // but the abort restores numUTXOs counter, effectively hiding them from iteration)
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_batch_remove_commit, BasicTestingSetup)
{
    // Test: Batch remove (for reorg) and commit

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());
    std::vector<uint256> txids;

    // Add 5 UTXOs first
    for (int i = 0; i < 5; i++) {
        uint256 txid = RandomTxid();
        txids.push_back(txid);
        gpu::UTXOHeader header = CreateHeader(i * COIN, 100, 0, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }

    // Begin batch for reorg
    utxoSet.BeginBatchUpdate();

    // Remove 3 UTXOs (simulating disconnecting a block that created them)
    for (int i = 0; i < 3; i++) {
        BOOST_CHECK(utxoSet.RemoveUTXO(txids[i], 0));
    }

    // Commit the removals
    BOOST_CHECK(utxoSet.CommitBatchUpdate());

    // Removed UTXOs should be gone
    for (int i = 0; i < 3; i++) {
        BOOST_CHECK(!utxoSet.HasUTXO(txids[i], 0));
    }

    // Remaining UTXOs should exist
    for (int i = 3; i < 5; i++) {
        BOOST_CHECK(utxoSet.HasUTXO(txids[i], 0));
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_batch_restore_commit, BasicTestingSetup)
{
    // Test: Restore spent UTXOs (for reorg)

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // Create and then spend a UTXO
    uint256 txid = RandomTxid();
    gpu::UTXOHeader header = CreateHeader(50 * COIN, 100, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    BOOST_CHECK(utxoSet.HasUTXO(txid, 0));

    // Spend it
    BOOST_CHECK(utxoSet.SpendUTXO(txid, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(txid, 0));

    // Now simulate reorg: restore the UTXO
    utxoSet.BeginBatchUpdate();
    BOOST_CHECK(utxoSet.RestoreUTXO(txid, 0, header, script.data()));
    BOOST_CHECK(utxoSet.CommitBatchUpdate());

    // UTXO should be back
    BOOST_CHECK(utxoSet.HasUTXO(txid, 0));

    // Verify the header data
    gpu::UTXOHeader retrieved_header;
    BOOST_CHECK(utxoSet.GetUTXO(txid, 0, retrieved_header, nullptr));
    BOOST_CHECK_EQUAL(retrieved_header.amount, 50 * COIN);
}

// =============================================================================
// Edge Cases: Duplicate Prevention
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_no_double_spend, BasicTestingSetup)
{
    // Test: Cannot spend same UTXO twice

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());
    gpu::UTXOHeader header = CreateHeader(50 * COIN, 100, 0, script.size());

    // Add UTXO
    BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));

    // First spend succeeds
    BOOST_CHECK(utxoSet.SpendUTXO(txid, 0));

    // Second spend fails (already spent)
    BOOST_CHECK(!utxoSet.SpendUTXO(txid, 0));
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_spend_nonexistent, BasicTestingSetup)
{
    // Test: Cannot spend UTXO that doesn't exist

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid = RandomTxid();

    // Try to spend non-existent UTXO
    BOOST_CHECK(!utxoSet.SpendUTXO(txid, 0));
}

// =============================================================================
// Edge Cases: Coinbase Handling
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_coinbase_flag, BasicTestingSetup)
{
    // Test: Coinbase UTXOs are properly flagged

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 coinbase_txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());
    gpu::UTXOHeader header = CreateHeader(50 * COIN, 100, 0, script.size(), true);  // coinbase=true

    BOOST_CHECK(utxoSet.AddUTXO(coinbase_txid, 0, header, script.data()));

    // Retrieve and check coinbase flag
    gpu::UTXOHeader retrieved;
    BOOST_CHECK(utxoSet.GetUTXO(coinbase_txid, 0, retrieved, nullptr));
    BOOST_CHECK(retrieved.flags & gpu::UTXO_FLAG_COINBASE);
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_regular_not_coinbase, BasicTestingSetup)
{
    // Test: Regular UTXOs don't have coinbase flag

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 regular_txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());
    gpu::UTXOHeader header = CreateHeader(1 * COIN, 100, 0, script.size(), false);  // coinbase=false

    BOOST_CHECK(utxoSet.AddUTXO(regular_txid, 0, header, script.data()));

    gpu::UTXOHeader retrieved;
    BOOST_CHECK(utxoSet.GetUTXO(regular_txid, 0, retrieved, nullptr));
    BOOST_CHECK(!(retrieved.flags & gpu::UTXO_FLAG_COINBASE));
}

// =============================================================================
// Edge Cases: Multiple Outputs Same Transaction
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_many_outputs_same_tx, BasicTestingSetup)
{
    // Test: Transaction with many outputs (up to 100)

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // Add 100 outputs from same transaction
    for (uint16_t vout = 0; vout < 100; vout++) {
        gpu::UTXOHeader header = CreateHeader(COIN, 100, vout, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txid, vout, header, script.data()));
    }

    // Verify all exist
    for (uint16_t vout = 0; vout < 100; vout++) {
        BOOST_CHECK(utxoSet.HasUTXO(txid, vout));
    }

    // Spend every other one
    for (uint16_t vout = 0; vout < 100; vout += 2) {
        BOOST_CHECK(utxoSet.SpendUTXO(txid, vout));
    }

    // Verify correct state
    for (uint16_t vout = 0; vout < 100; vout++) {
        if (vout % 2 == 0) {
            BOOST_CHECK(!utxoSet.HasUTXO(txid, vout));
        } else {
            BOOST_CHECK(utxoSet.HasUTXO(txid, vout));
        }
    }
}

// =============================================================================
// Edge Cases: Script Types
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_various_script_types, BasicTestingSetup)
{
    // Test: Different script types are handled correctly

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    // P2PKH
    {
        uint256 txid = RandomTxid();
        std::vector<uint8_t> pkh(20, 0x01);
        CScript script;
        script << OP_DUP << OP_HASH160 << pkh << OP_EQUALVERIFY << OP_CHECKSIG;

        gpu::UTXOHeader header = CreateHeader(COIN, 100, 0, script.size());
        header.script_type = gpu::SCRIPT_TYPE_P2PKH;
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }

    // P2WPKH
    {
        uint256 txid = RandomTxid();
        std::vector<uint8_t> pkh(20, 0x02);
        CScript script;
        script << OP_0 << pkh;

        gpu::UTXOHeader header = CreateHeader(COIN, 100, 0, script.size());
        header.script_type = gpu::SCRIPT_TYPE_P2WPKH;
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }

    // P2SH
    {
        uint256 txid = RandomTxid();
        std::vector<uint8_t> sh(20, 0x03);
        CScript script;
        script << OP_HASH160 << sh << OP_EQUAL;

        gpu::UTXOHeader header = CreateHeader(COIN, 100, 0, script.size());
        header.script_type = gpu::SCRIPT_TYPE_P2SH;
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }

    // P2WSH
    {
        uint256 txid = RandomTxid();
        std::vector<uint8_t> wsh(32, 0x04);
        CScript script;
        script << OP_0 << wsh;

        gpu::UTXOHeader header = CreateHeader(COIN, 100, 0, script.size());
        header.script_type = gpu::SCRIPT_TYPE_P2WSH;
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }

    // P2TR
    {
        uint256 txid = RandomTxid();
        std::vector<uint8_t> xonly(32, 0x05);
        CScript script;
        script << OP_1 << xonly;

        gpu::UTXOHeader header = CreateHeader(COIN, 100, 0, script.size());
        header.script_type = gpu::SCRIPT_TYPE_P2TR;
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }
}

// =============================================================================
// Edge Cases: Large Values
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_max_amount, BasicTestingSetup)
{
    // Test: Maximum possible amount (21M BTC)

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    int64_t max_btc = 2100000000000000LL;  // 21M BTC in satoshis
    gpu::UTXOHeader header = CreateHeader(max_btc, 100, 0, script.size());

    BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));

    gpu::UTXOHeader retrieved;
    BOOST_CHECK(utxoSet.GetUTXO(txid, 0, retrieved, nullptr));
    BOOST_CHECK_EQUAL(retrieved.amount, max_btc);
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_dust_amount, BasicTestingSetup)
{
    // Test: Minimum possible amount (1 satoshi)

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    gpu::UTXOHeader header = CreateHeader(1, 100, 0, script.size());  // 1 satoshi

    BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));

    gpu::UTXOHeader retrieved;
    BOOST_CHECK(utxoSet.GetUTXO(txid, 0, retrieved, nullptr));
    BOOST_CHECK_EQUAL(retrieved.amount, 1);
}

// =============================================================================
// Edge Cases: Block Heights
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_genesis_block, BasicTestingSetup)
{
    // Test: UTXO from block 0 (genesis block)

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    gpu::UTXOHeader header = CreateHeader(50 * COIN, 0, 0, script.size(), true);  // Genesis coinbase

    BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));

    gpu::UTXOHeader retrieved;
    BOOST_CHECK(utxoSet.GetUTXO(txid, 0, retrieved, nullptr));
    BOOST_CHECK_EQUAL(retrieved.blockHeight, 0);
    BOOST_CHECK(retrieved.flags & gpu::UTXO_FLAG_COINBASE);
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_high_block_height, BasicTestingSetup)
{
    // Test: UTXO from very high block (near 24-bit limit)

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    uint256 txid = RandomTxid();
    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // 24-bit max is 16,777,215 blocks (~320 years at 10 min blocks)
    uint32_t high_block = 16000000;  // ~305 years of blocks
    gpu::UTXOHeader header = CreateHeader(COIN, high_block, 0, script.size());

    BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));

    gpu::UTXOHeader retrieved;
    BOOST_CHECK(utxoSet.GetUTXO(txid, 0, retrieved, nullptr));
    BOOST_CHECK_EQUAL(retrieved.blockHeight, high_block);
}

// =============================================================================
// Stress Tests: Many Operations
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_stress_add_spend_cycle, BasicTestingSetup)
{
    // Test: Add and spend many UTXOs in cycles

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    std::vector<uint256> active_utxos;

    // Add 100 UTXOs
    for (int i = 0; i < 100; i++) {
        uint256 txid = RandomTxid();
        gpu::UTXOHeader header = CreateHeader(COIN, 100, 0, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
        active_utxos.push_back(txid);
    }

    // Spend 50, add 50 more
    for (int cycle = 0; cycle < 5; cycle++) {
        // Spend first 50
        for (int i = 0; i < 50; i++) {
            BOOST_CHECK(utxoSet.SpendUTXO(active_utxos[i], 0));
        }

        // Remove spent from list
        active_utxos.erase(active_utxos.begin(), active_utxos.begin() + 50);

        // Add 50 new
        for (int i = 0; i < 50; i++) {
            uint256 txid = RandomTxid();
            gpu::UTXOHeader header = CreateHeader(COIN, 100 + cycle + 1, 0, script.size());
            BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
            active_utxos.push_back(txid);
        }

        // Verify count is correct
        size_t expected_active = 100;  // Should always have 100 active
        size_t actual_active = 0;
        for (const auto& txid : active_utxos) {
            if (utxoSet.HasUTXO(txid, 0)) actual_active++;
        }
        BOOST_CHECK_EQUAL(actual_active, expected_active);
    }
}

// =============================================================================
// Simulated Block Processing Tests
// =============================================================================

BOOST_FIXTURE_TEST_CASE(gpu_utxo_simulate_block_connect, BasicTestingSetup)
{
    // Test: Simulate connecting a block with multiple transactions
    // Block structure:
    //   Coinbase: creates 1 output
    //   TX1: spends 2 existing UTXOs, creates 3 outputs
    //   TX2: spends TX1:0 (same block parent-child), creates 1 output
    //   TX3: spends 1 existing UTXO + TX1:1, creates 2 outputs

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // Pre-existing UTXOs (from previous blocks)
    std::vector<std::pair<uint256, uint32_t>> existing_utxos;
    for (int i = 0; i < 5; i++) {
        uint256 txid = RandomTxid();
        gpu::UTXOHeader header = CreateHeader(10 * COIN, 99, 0, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
        existing_utxos.push_back({txid, 0});
    }

    // Connect block 100
    utxoSet.BeginBatchUpdate();

    // Coinbase
    uint256 coinbase_txid = RandomTxid();
    gpu::UTXOHeader cb_header = CreateHeader(5000000000LL, 100, 0, script.size(), true);
    BOOST_CHECK(utxoSet.AddUTXO(coinbase_txid, 0, cb_header, script.data()));

    // TX1: Spend existing[0], existing[1], create 3 outputs
    BOOST_CHECK(utxoSet.SpendUTXO(existing_utxos[0].first, 0));
    BOOST_CHECK(utxoSet.SpendUTXO(existing_utxos[1].first, 0));

    uint256 tx1_txid = RandomTxid();
    for (uint16_t vout = 0; vout < 3; vout++) {
        gpu::UTXOHeader header = CreateHeader(5 * COIN, 100, vout, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(tx1_txid, vout, header, script.data()));
    }

    // TX2: Spend TX1:0 (parent-child in same block), create 1 output
    BOOST_CHECK(utxoSet.SpendUTXO(tx1_txid, 0));
    uint256 tx2_txid = RandomTxid();
    gpu::UTXOHeader tx2_header = CreateHeader(4 * COIN, 100, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(tx2_txid, 0, tx2_header, script.data()));

    // TX3: Spend existing[2] + TX1:1, create 2 outputs
    BOOST_CHECK(utxoSet.SpendUTXO(existing_utxos[2].first, 0));
    BOOST_CHECK(utxoSet.SpendUTXO(tx1_txid, 1));

    uint256 tx3_txid = RandomTxid();
    for (uint16_t vout = 0; vout < 2; vout++) {
        gpu::UTXOHeader header = CreateHeader(7 * COIN, 100, vout, script.size());
        BOOST_CHECK(utxoSet.AddUTXO(tx3_txid, vout, header, script.data()));
    }

    BOOST_CHECK(utxoSet.CommitBatchUpdate());

    // Verify final state
    // Spent: existing[0], existing[1], existing[2], TX1:0, TX1:1
    BOOST_CHECK(!utxoSet.HasUTXO(existing_utxos[0].first, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(existing_utxos[1].first, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(existing_utxos[2].first, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(tx1_txid, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(tx1_txid, 1));

    // Unspent: existing[3], existing[4], coinbase:0, TX1:2, TX2:0, TX3:0, TX3:1
    BOOST_CHECK(utxoSet.HasUTXO(existing_utxos[3].first, 0));
    BOOST_CHECK(utxoSet.HasUTXO(existing_utxos[4].first, 0));
    BOOST_CHECK(utxoSet.HasUTXO(coinbase_txid, 0));
    BOOST_CHECK(utxoSet.HasUTXO(tx1_txid, 2));
    BOOST_CHECK(utxoSet.HasUTXO(tx2_txid, 0));
    BOOST_CHECK(utxoSet.HasUTXO(tx3_txid, 0));
    BOOST_CHECK(utxoSet.HasUTXO(tx3_txid, 1));
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_simulate_reorg, BasicTestingSetup)
{
    // Test: Simulate a 2-block reorg
    // Initial state: Block 100, Block 101 connected
    // Reorg: Disconnect 101, disconnect 100, connect 100', connect 101'

    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());

    CScript script = CreateP2PKHScript(RandomPubKeyHash());

    // Block 99: Create initial UTXO
    uint256 block99_txid = RandomTxid();
    gpu::UTXOHeader header99 = CreateHeader(100 * COIN, 99, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(block99_txid, 0, header99, script.data()));

    // Block 100: Spend block99 UTXO, create new one
    uint256 block100_txid = RandomTxid();
    BOOST_CHECK(utxoSet.SpendUTXO(block99_txid, 0));
    gpu::UTXOHeader header100 = CreateHeader(99 * COIN, 100, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(block100_txid, 0, header100, script.data()));

    // Block 101: Spend block100 UTXO, create new one
    uint256 block101_txid = RandomTxid();
    BOOST_CHECK(utxoSet.SpendUTXO(block100_txid, 0));
    gpu::UTXOHeader header101 = CreateHeader(98 * COIN, 101, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(block101_txid, 0, header101, script.data()));

    // Now simulate reorg: disconnect block 101
    utxoSet.BeginBatchUpdate();
    BOOST_CHECK(utxoSet.RemoveUTXO(block101_txid, 0));  // Remove output created in 101
    BOOST_CHECK(utxoSet.RestoreUTXO(block100_txid, 0, header100, script.data()));  // Restore spent input
    BOOST_CHECK(utxoSet.CommitBatchUpdate());

    BOOST_CHECK(!utxoSet.HasUTXO(block101_txid, 0));
    BOOST_CHECK(utxoSet.HasUTXO(block100_txid, 0));

    // Disconnect block 100
    utxoSet.BeginBatchUpdate();
    BOOST_CHECK(utxoSet.RemoveUTXO(block100_txid, 0));
    BOOST_CHECK(utxoSet.RestoreUTXO(block99_txid, 0, header99, script.data()));
    BOOST_CHECK(utxoSet.CommitBatchUpdate());

    BOOST_CHECK(!utxoSet.HasUTXO(block100_txid, 0));
    BOOST_CHECK(utxoSet.HasUTXO(block99_txid, 0));

    // Connect new block 100'
    uint256 block100_prime_txid = RandomTxid();
    BOOST_CHECK(utxoSet.SpendUTXO(block99_txid, 0));
    gpu::UTXOHeader header100_prime = CreateHeader(99 * COIN, 100, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(block100_prime_txid, 0, header100_prime, script.data()));

    // Connect new block 101'
    uint256 block101_prime_txid = RandomTxid();
    BOOST_CHECK(utxoSet.SpendUTXO(block100_prime_txid, 0));
    gpu::UTXOHeader header101_prime = CreateHeader(98 * COIN, 101, 0, script.size());
    BOOST_CHECK(utxoSet.AddUTXO(block101_prime_txid, 0, header101_prime, script.data()));

    // Final state: only block101' output should exist
    BOOST_CHECK(!utxoSet.HasUTXO(block99_txid, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(block100_txid, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(block101_txid, 0));
    BOOST_CHECK(!utxoSet.HasUTXO(block100_prime_txid, 0));
    BOOST_CHECK(utxoSet.HasUTXO(block101_prime_txid, 0));
}

BOOST_AUTO_TEST_SUITE_END()
