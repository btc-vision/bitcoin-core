// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <coins.h>
#include <consensus/amount.h>
#include <primitives/transaction.h>
#include <script/script.h>
#include <uint256.h>
#include <validation.h>
#include <txdb.h>
#include <random.h>
#include <node/coins_view_args.h>

#include "../gpu_kernel/gpu_utxo.h"
#include "../gpu_kernel/gpu_types.h"

#include <memory>
#include <vector>
#include <unordered_set>

// Forward declaration of the GPU loader function
namespace gpu {
    bool LoadUTXOSetToCPU(const void* coinsCache, 
                          std::vector<UTXOHeader>& headers,
                          std::vector<uint256_gpu>& uniqueTxids,
                          std::vector<uint8_t>& scriptBlob,
                          size_t maxUTXOs,
                          size_t maxScriptBlobSize);
}

BOOST_FIXTURE_TEST_SUITE(gpu_utxo_loader_tests, TestingSetup)

// Helper function to create an in-memory CCoinsViewDB
static std::unique_ptr<CCoinsViewDB> CreateTestCoinsViewDB() {
    DBParams db_params{
        .path = "",  // Empty path for in-memory
        .cache_bytes = 10 << 20,  // 10MB
        .memory_only = true,
        .wipe_data = true,
    };
    CoinsViewOptions options;
    // Use default options
    return std::make_unique<CCoinsViewDB>(db_params, options);
}

// Helper function to create a dummy transaction output
static CTxOut CreateDummyOutput(CAmount value, const CScript& script) {
    return CTxOut(value, script);
}

// Helper function to create a random outpoint
static COutPoint RandomOutPoint(FastRandomContext& rng) {
    return COutPoint(Txid::FromUint256(rng.rand256()), rng.rand32());
}

// Helper function to create a random coin
static Coin RandomCoin(FastRandomContext& rng, CAmount value, int height) {
    // Create a simple P2PKH script
    CScript script;
    script << OP_DUP << OP_HASH160 << rng.randbytes(20) << OP_EQUALVERIFY << OP_CHECKSIG;
    
    CTxOut output(value, script);
    return Coin(std::move(output), height, false);
}

BOOST_AUTO_TEST_CASE(test_empty_utxo_set)
{
    // Create an empty coins view
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Test loading from empty set
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        1000000,  // maxUTXOs
        100 << 20 // 100MB script blob
    );
    
    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(headers.size(), 0);
    BOOST_CHECK_EQUAL(uniqueTxids.size(), 0);
    BOOST_CHECK_EQUAL(scriptBlob.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_database_only_utxos)
{
    FastRandomContext rng(42);
    
    // Create a database with some UTXOs
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Set a valid best block hash
    cache.SetBestBlock(rng.rand256());
    
    // Add UTXOs to the database
    std::vector<COutPoint> outpoints;
    std::vector<Coin> coins;
    const int num_utxos = 10;
    
    for (int i = 0; i < num_utxos; i++) {
        COutPoint outpoint = RandomOutPoint(rng);
        Coin coin = RandomCoin(rng, (i + 1) * COIN, 100 + i);
        
        outpoints.push_back(outpoint);
        coins.push_back(coin);
        
        cache.AddCoin(outpoint, std::move(coin), false);
    }
    
    // Flush to database
    BOOST_CHECK(cache.Flush());
    
    // Create a new cache on top
    CCoinsViewCache cache2(base.get());
    
    // Load UTXOs
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache2, headers, uniqueTxids, scriptBlob,
        1000000,
        100 << 20
    );
    
    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(headers.size(), num_utxos);
    BOOST_CHECK_GE(uniqueTxids.size(), 1); // At least one unique txid
    BOOST_CHECK_GT(scriptBlob.size(), 0);  // Scripts were stored
    
    // Verify amounts are present (order not guaranteed from database)
    std::unordered_set<CAmount> expected_amounts;
    for (const auto& coin : coins) {
        expected_amounts.insert(coin.out.nValue);
    }
    
    std::unordered_set<CAmount> actual_amounts;
    for (const auto& header : headers) {
        actual_amounts.insert(header.amount);
    }
    
    BOOST_CHECK(expected_amounts == actual_amounts);
}

BOOST_AUTO_TEST_CASE(test_cache_only_utxos)
{
    FastRandomContext rng(43);
    
    // Create an empty database
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Add UTXOs only to cache (not flushed)
    std::vector<COutPoint> outpoints;
    std::vector<Coin> coins;
    const int num_utxos = 5;
    
    for (int i = 0; i < num_utxos; i++) {
        COutPoint outpoint = RandomOutPoint(rng);
        Coin coin = RandomCoin(rng, (i + 1) * COIN, 200 + i);
        
        outpoints.push_back(outpoint);
        coins.push_back(coin);
        
        cache.AddCoin(outpoint, std::move(coin), false);
    }
    
    // DO NOT flush - keep in cache only
    
    // Load UTXOs (should get cache-only entries)
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        1000000,
        100 << 20
    );
    
    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(headers.size(), num_utxos);
    BOOST_CHECK_GE(uniqueTxids.size(), 1);
    
    // Verify the UTXOs match what we added (order not guaranteed)
    std::unordered_set<CAmount> expected_amounts;
    std::unordered_set<uint32_t> expected_heights;
    for (const auto& coin : coins) {
        expected_amounts.insert(coin.out.nValue);
        expected_heights.insert(coin.nHeight & 0xFFFFFF);
    }
    
    std::unordered_set<CAmount> actual_amounts;
    std::unordered_set<uint32_t> actual_heights;
    for (const auto& header : headers) {
        actual_amounts.insert(header.amount);
        actual_heights.insert(header.blockHeight);
    }
    
    BOOST_CHECK(expected_amounts == actual_amounts);
    BOOST_CHECK(expected_heights == actual_heights);
}

BOOST_AUTO_TEST_CASE(test_mixed_database_and_cache)
{
    FastRandomContext rng(44);
    
    // Create database with some UTXOs
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Set a valid best block hash
    cache.SetBestBlock(rng.rand256());
    
    // Add some UTXOs to database
    const int num_db_utxos = 5;
    std::vector<Coin> db_coins;
    
    for (int i = 0; i < num_db_utxos; i++) {
        COutPoint outpoint = RandomOutPoint(rng);
        Coin coin = RandomCoin(rng, (i + 1) * COIN, 300 + i);
        db_coins.push_back(coin);
        cache.AddCoin(outpoint, std::move(coin), false);
    }
    
    // Flush to database
    BOOST_CHECK(cache.Flush());
    
    // Add more UTXOs to cache only
    const int num_cache_utxos = 3;
    std::vector<Coin> cache_coins;
    
    for (int i = 0; i < num_cache_utxos; i++) {
        COutPoint outpoint = RandomOutPoint(rng);
        Coin coin = RandomCoin(rng, (i + 10) * COIN, 400 + i);
        cache_coins.push_back(coin);
        cache.AddCoin(outpoint, std::move(coin), false);
    }
    
    // Load all UTXOs
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        1000000,
        100 << 20
    );
    
    BOOST_CHECK(result);
    // Should get both database and cache UTXOs
    BOOST_CHECK_EQUAL(headers.size(), num_db_utxos + num_cache_utxos);
}

BOOST_AUTO_TEST_CASE(test_modified_utxo_deduplication)
{
    FastRandomContext rng(45);
    
    // Create database with a UTXO
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Set a valid best block hash
    cache.SetBestBlock(rng.rand256());
    
    COutPoint outpoint = RandomOutPoint(rng);
    Coin original_coin = RandomCoin(rng, 5 * COIN, 100);
    CAmount original_amount = original_coin.out.nValue;
    
    cache.AddCoin(outpoint, Coin(original_coin), false);
    BOOST_CHECK(cache.Flush());
    
    // Modify the coin in cache (simulate spending part of it)
    // TODO(human): Implement the logic to modify a UTXO in the cache
    // This would test that we get the cache version, not the database version
    // Consider: How should we handle modified UTXOs? Should we update the amount?
    
    // For now, add a different coin to the same outpoint (overwrite)
    Coin modified_coin = RandomCoin(rng, 3 * COIN, 100);
    cache.AddCoin(outpoint, std::move(modified_coin), true); // allow overwrite
    
    // Load UTXOs
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        1000000,
        100 << 20
    );
    
    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(headers.size(), 1);
    // Should get the modified amount from cache, not original
    BOOST_CHECK_EQUAL(headers[0].amount, 3 * COIN);
    BOOST_CHECK_NE(headers[0].amount, original_amount);
}

BOOST_AUTO_TEST_CASE(test_spent_coin_filtering)
{
    FastRandomContext rng(46);
    
    // Create database with UTXOs
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Set a valid best block hash
    cache.SetBestBlock(rng.rand256());
    
    // Add some UTXOs
    std::vector<COutPoint> outpoints;
    for (int i = 0; i < 5; i++) {
        outpoints.push_back(RandomOutPoint(rng));
        cache.AddCoin(outpoints[i], RandomCoin(rng, (i + 1) * COIN, 100), false);
    }
    
    BOOST_CHECK(cache.Flush());
    
    // Spend some coins in cache
    BOOST_CHECK(cache.SpendCoin(outpoints[1]));
    BOOST_CHECK(cache.SpendCoin(outpoints[3]));
    
    // Verify spent status
    for (int i = 0; i < 5; i++) {
        auto coin = cache.GetCoin(outpoints[i]);
        if (i == 1 || i == 3) {
            BOOST_CHECK(!coin.has_value() || coin->IsSpent());
        } else {
            BOOST_CHECK(coin.has_value() && !coin->IsSpent());
        }
    }
    
    // Load UTXOs
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        1000000,
        100 << 20
    );
    
    BOOST_CHECK(result);
    // Should only get unspent coins (3 out of 5)
    BOOST_CHECK_EQUAL(headers.size(), 3);
}

BOOST_AUTO_TEST_CASE(test_max_utxo_limit)
{
    FastRandomContext rng(47);
    
    // Create database with many UTXOs
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Set a valid best block hash
    cache.SetBestBlock(rng.rand256());
    
    // Add more UTXOs than the limit
    const int num_utxos = 10;
    const size_t max_utxos = 5;
    
    for (int i = 0; i < num_utxos; i++) {
        COutPoint outpoint = RandomOutPoint(rng);
        cache.AddCoin(outpoint, RandomCoin(rng, COIN, 100), false);
    }
    
    BOOST_CHECK(cache.Flush());
    
    // Load with a limit
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        max_utxos,  // Limited to 5
        100 << 20
    );
    
    BOOST_CHECK(result);
    // Should respect the limit
    BOOST_CHECK_EQUAL(headers.size(), max_utxos);
}

BOOST_AUTO_TEST_CASE(test_script_blob_limit)
{
    FastRandomContext rng(48);
    
    // Create database with UTXOs that have large scripts
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Set a valid best block hash
    cache.SetBestBlock(rng.rand256());
    
    // Create UTXOs with large scripts
    const size_t script_size = 1000;  // 1KB per script
    const size_t max_blob_size = 2500; // Only room for ~2 scripts
    
    for (int i = 0; i < 5; i++) {
        COutPoint outpoint = RandomOutPoint(rng);
        
        // Create a large script
        CScript large_script;
        large_script << rng.randbytes(script_size - 10); // Leave room for opcodes
        
        CTxOut output(COIN, large_script);
        Coin coin(std::move(output), 100, false);
        
        cache.AddCoin(outpoint, std::move(coin), false);
    }
    
    BOOST_CHECK(cache.Flush());
    
    // Load with script blob limit
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        1000000,
        max_blob_size
    );
    
    BOOST_CHECK(result);
    // Should stop when script blob is full
    BOOST_CHECK_LT(headers.size(), 5);
    BOOST_CHECK_LE(scriptBlob.size(), max_blob_size);
}

BOOST_AUTO_TEST_CASE(test_txid_deduplication)
{
    FastRandomContext rng(49);
    
    // Create database with multiple UTXOs from same transaction
    auto base = CreateTestCoinsViewDB();
    CCoinsViewCache cache(base.get());
    
    // Set a valid best block hash
    cache.SetBestBlock(rng.rand256());
    
    // Use same txid for multiple outputs
    Txid common_txid = Txid::FromUint256(rng.rand256());
    
    for (uint32_t vout = 0; vout < 5; vout++) {
        COutPoint outpoint(common_txid, vout);
        cache.AddCoin(outpoint, RandomCoin(rng, COIN, 100), false);
    }
    
    // Add some UTXOs from different txids
    for (int i = 0; i < 3; i++) {
        COutPoint outpoint = RandomOutPoint(rng);
        cache.AddCoin(outpoint, RandomCoin(rng, COIN, 100), false);
    }
    
    BOOST_CHECK(cache.Flush());
    
    // Load UTXOs
    std::vector<gpu::UTXOHeader> headers;
    std::vector<gpu::uint256_gpu> uniqueTxids;
    std::vector<uint8_t> scriptBlob;
    
    bool result = gpu::LoadUTXOSetToCPU(
        &cache, headers, uniqueTxids, scriptBlob,
        1000000,
        100 << 20
    );
    
    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(headers.size(), 8); // 5 + 3 UTXOs
    // Should have deduplicated txids (1 common + up to 3 different)
    BOOST_CHECK_LE(uniqueTxids.size(), 4);
    BOOST_CHECK_GE(uniqueTxids.size(), 2); // At least the common one and one other
}

BOOST_AUTO_TEST_SUITE_END()