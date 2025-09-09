// Copyright (c) 2024-present The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_utxo.h>
#include <gpu_kernel/gpu_logging.h>
#include <coins.h>
#include <primitives/transaction.h>
#include <random.h>
#include <util/time.h>

#include <vector>
#include <memory>

BOOST_AUTO_TEST_SUITE(gpu_utxo_tests)

BOOST_FIXTURE_TEST_CASE(gpu_utxo_initialization, BasicTestingSetup)
{
    // Test GPU UTXO set initialization
    gpu::GPUUTXOSet utxoSet;
    
    // Initialize with default memory limits
    bool result = utxoSet.Initialize();
    BOOST_CHECK(result);
    
    // Check that memory was allocated
    BOOST_CHECK_GT(utxoSet.GetMaxUTXOs(), 0);
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 0);
    BOOST_CHECK_EQUAL(utxoSet.GetLoadFactor(), 0);
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_add_and_query, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Create a test UTXO
    uint256 txid = GetRandHash();
    uint32_t vout = 0;
    
    gpu::UTXOHeader header;
    memset(&header, 0, sizeof(header));
    header.amount = 50 * COIN;
    header.blockHeight = 100;
    header.txid_index = 0;  // First unique txid
    header.vout = vout;
    header.flags = 0;
    header.script_size = 25;  // P2PKH script size
    header.script_type = gpu::SCRIPT_TYPE_P2PKH;
    header.script_offset = 0;
    
    // Create a dummy P2PKH script
    uint8_t script[25] = {
        0x76, 0xa9, 0x14,  // OP_DUP OP_HASH160 <20>
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13,  // 20 bytes of pubkey hash
        0x88, 0xac  // OP_EQUALVERIFY OP_CHECKSIG
    };
    
    // Add the UTXO
    bool added = utxoSet.AddUTXO(txid, vout, header, script);
    BOOST_CHECK(added);
    
    // Query the UTXO
    bool exists = utxoSet.HasUTXO(txid, vout);
    BOOST_CHECK(exists);
    
    // Check load factor increased
    BOOST_CHECK_GT(utxoSet.GetLoadFactor(), 0);
    BOOST_CHECK_EQUAL(utxoSet.GetNumUTXOs(), 1);
}

BOOST_FIXTURE_TEST_CASE(gpu_utxo_spend, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add a UTXO
    uint256 txid = GetRandHash();
    uint32_t vout = 0;
    
    gpu::UTXOHeader header;
    memset(&header, 0, sizeof(header));
    header.amount = 100 * COIN;
    header.blockHeight = 200;
    header.txid_index = 0;
    header.vout = vout;
    header.flags = 0;
    header.script_size = 0;  // No script for simplicity
    header.script_type = gpu::SCRIPT_TYPE_P2PKH;
    header.script_offset = 0;
    
    BOOST_REQUIRE(utxoSet.AddUTXO(txid, vout, header, nullptr));
    BOOST_REQUIRE(utxoSet.HasUTXO(txid, vout));
    
    // Spend the UTXO
    bool spent = utxoSet.SpendUTXO(txid, vout);
    BOOST_CHECK(spent);
    
    // Verify it's no longer available
    bool exists = utxoSet.HasUTXO(txid, vout);
    BOOST_CHECK(!exists);
}

BOOST_AUTO_TEST_SUITE_END()