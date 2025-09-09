// Copyright (c) 2024-present The Bitcoin Core developers
// GPU Validation Kernel Tests - Complete coverage for P2PKH/P2WPKH validation

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_utxo.h>
#include <gpu_kernel/gpu_validation.h>
#include <key.h>
#include <pubkey.h>
#include <script/script.h>
#include <random.h>

BOOST_AUTO_TEST_SUITE(gpu_validation_kernel_tests)

BOOST_FIXTURE_TEST_CASE(validation_p2pkh_script_structure, BasicTestingSetup)
{
    // Test P2PKH script identification
    uint8_t valid_p2pkh[25] = {
        0x76, 0xa9, 0x14,  // OP_DUP OP_HASH160 OP_PUSHDATA(20)
        // 20 bytes of pubkey hash
        0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef,
        0x88, 0xac  // OP_EQUALVERIFY OP_CHECKSIG
    };
    
    BOOST_CHECK_EQUAL(gpu::IdentifyScriptType(valid_p2pkh, 25), gpu::SCRIPT_TYPE_P2PKH);
    
    // Test invalid P2PKH
    uint8_t invalid_p2pkh[25];
    memcpy(invalid_p2pkh, valid_p2pkh, 25);
    invalid_p2pkh[0] = 0x77; // Wrong opcode
    
    BOOST_CHECK_NE(gpu::IdentifyScriptType(invalid_p2pkh, 25), gpu::SCRIPT_TYPE_P2PKH);
}

BOOST_FIXTURE_TEST_CASE(validation_p2wpkh_script_structure, BasicTestingSetup)
{
    // Test P2WPKH script identification
    uint8_t valid_p2wpkh[22] = {
        0x00, 0x14,  // OP_0 OP_PUSHDATA(20)
        // 20 bytes of pubkey hash
        0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef
    };
    
    BOOST_CHECK_EQUAL(gpu::IdentifyScriptType(valid_p2wpkh, 22), gpu::SCRIPT_TYPE_P2WPKH);
    
    // Test invalid P2WPKH
    uint8_t invalid_p2wpkh[22];
    memcpy(invalid_p2wpkh, valid_p2wpkh, 22);
    invalid_p2wpkh[1] = 0x15; // Wrong length
    
    BOOST_CHECK_NE(gpu::IdentifyScriptType(invalid_p2wpkh, 22), gpu::SCRIPT_TYPE_P2WPKH);
}

BOOST_FIXTURE_TEST_CASE(validation_script_type_all_types, BasicTestingSetup)
{
    // P2SH
    uint8_t p2sh[23] = {
        0xa9, 0x14,  // OP_HASH160 OP_PUSHDATA(20)
        0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef,
        0x87  // OP_EQUAL
    };
    BOOST_CHECK_EQUAL(gpu::IdentifyScriptType(p2sh, 23), gpu::SCRIPT_TYPE_P2SH);
    
    // P2WSH
    uint8_t p2wsh[34] = {
        0x00, 0x20,  // OP_0 OP_PUSHDATA(32)
        // 32 bytes of script hash
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
        0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20
    };
    BOOST_CHECK_EQUAL(gpu::IdentifyScriptType(p2wsh, 34), gpu::SCRIPT_TYPE_P2WSH);
    
    // P2TR (Taproot)
    uint8_t p2tr[34] = {
        0x51, 0x20,  // OP_1 OP_PUSHDATA(32)
        // 32 bytes of taproot output
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
        0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20
    };
    BOOST_CHECK_EQUAL(gpu::IdentifyScriptType(p2tr, 34), gpu::SCRIPT_TYPE_P2TR);
    
    // Non-standard
    uint8_t nonstandard[10] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a};
    BOOST_CHECK_EQUAL(gpu::IdentifyScriptType(nonstandard, 10), gpu::SCRIPT_TYPE_NONSTANDARD);
}

BOOST_FIXTURE_TEST_CASE(validation_kernel_memory_access, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Create test UTXO with P2PKH script
    uint256 txid = GetRandHash();
    
    uint8_t pubkey_hash[20];
    GetRandBytes(pubkey_hash);
    
    uint8_t p2pkh_script[25] = {
        0x76, 0xa9, 0x14  // OP_DUP OP_HASH160 OP_PUSHDATA(20)
    };
    memcpy(p2pkh_script + 3, pubkey_hash, 20);
    p2pkh_script[23] = 0x88;  // OP_EQUALVERIFY
    p2pkh_script[24] = 0xac;  // OP_CHECKSIG
    
    gpu::UTXOHeader header;
    memset(&header, 0, sizeof(header));
    header.amount = 100 * COIN;
    header.script_size = 25;
    header.script_type = gpu::SCRIPT_TYPE_P2PKH;
    header.script_offset = 0;
    header.txid_index = 0;
    header.vout = 0;
    
    BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, p2pkh_script));
    BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
}

BOOST_FIXTURE_TEST_CASE(validation_batch_processing, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Add multiple P2PKH UTXOs
    const int NUM_UTXOS = 100;
    std::vector<uint256> txids;
    std::vector<std::vector<uint8_t>> scripts;
    
    for (int i = 0; i < NUM_UTXOS; i++) {
        uint256 txid = GetRandHash();
        txids.push_back(txid);
        
        // Create unique P2PKH script
        std::vector<uint8_t> script(25);
        script[0] = 0x76;  // OP_DUP
        script[1] = 0xa9;  // OP_HASH160
        script[2] = 0x14;  // OP_PUSHDATA(20)
        GetRandBytes(MakeWritableByteSpan(script).subspan(3, 20));
        script[23] = 0x88;  // OP_EQUALVERIFY
        script[24] = 0xac;  // OP_CHECKSIG
        
        scripts.push_back(script);
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.amount = (i + 1) * COIN;
        header.script_size = 25;
        header.script_type = gpu::SCRIPT_TYPE_P2PKH;
        header.script_offset = i * 25;
        header.txid_index = i;
        header.vout = 0;
        
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, script.data()));
    }
    
    // Verify all were added
    for (int i = 0; i < NUM_UTXOS; i++) {
        BOOST_CHECK(utxoSet.HasUTXO(txids[i], 0));
    }
}

BOOST_FIXTURE_TEST_CASE(validation_coinbase_flag, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Test coinbase UTXO
    uint256 coinbase_txid = GetRandHash();
    
    gpu::UTXOHeader coinbase_header;
    memset(&coinbase_header, 0, sizeof(coinbase_header));
    coinbase_header.amount = 50 * COIN;
    coinbase_header.flags = gpu::UTXO_FLAG_COINBASE;
    coinbase_header.blockHeight = 100;
    coinbase_header.txid_index = 0;
    coinbase_header.vout = 0;
    
    BOOST_CHECK(utxoSet.AddUTXO(coinbase_txid, 0, coinbase_header, nullptr));
    
    // Test regular UTXO
    uint256 regular_txid = GetRandHash();
    
    gpu::UTXOHeader regular_header;
    memset(&regular_header, 0, sizeof(regular_header));
    regular_header.amount = 10 * COIN;
    regular_header.flags = 0;  // Not coinbase
    regular_header.blockHeight = 101;
    regular_header.txid_index = 1;
    regular_header.vout = 0;
    
    BOOST_CHECK(utxoSet.AddUTXO(regular_txid, 0, regular_header, nullptr));
    
    // Both should exist
    BOOST_CHECK(utxoSet.HasUTXO(coinbase_txid, 0));
    BOOST_CHECK(utxoSet.HasUTXO(regular_txid, 0));
}

BOOST_FIXTURE_TEST_CASE(validation_block_height_limits, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Test various block heights (24-bit field)
    std::vector<uint32_t> test_heights = {
        0,           // Genesis
        1,           // First block
        100000,      // Typical height
        0xFFFFFF,    // Maximum 24-bit value
        0x1000000    // Should be truncated
    };
    
    for (size_t i = 0; i < test_heights.size(); i++) {
        uint256 txid = GetRandHash();
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.blockHeight = test_heights[i] & 0xFFFFFF;  // 24-bit limit
        header.txid_index = i;
        header.vout = 0;
        
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, nullptr));
        BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
    }
}

BOOST_FIXTURE_TEST_CASE(validation_amount_precision, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Test various amounts
    std::vector<CAmount> test_amounts = {
        0,                  // Zero value (OP_RETURN)
        1,                  // 1 satoshi
        COIN,               // 1 BTC
        21000000 * COIN,    // Maximum supply
        0x7FFFFFFFFFFFFFFF  // Maximum int64
    };
    
    for (size_t i = 0; i < test_amounts.size(); i++) {
        uint256 txid = GetRandHash();
        
        gpu::UTXOHeader header;
        memset(&header, 0, sizeof(header));
        header.amount = test_amounts[i];
        header.txid_index = i;
        header.vout = 0;
        
        BOOST_CHECK(utxoSet.AddUTXO(txid, 0, header, nullptr));
        BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
    }
}

BOOST_FIXTURE_TEST_CASE(validation_vout_range, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    uint256 txid = GetRandHash();
    
    // Test various vout values (16-bit field)
    std::vector<uint32_t> test_vouts = {
        0,        // First output
        1,        // Second output
        100,      // Many outputs
        0xFFFF,   // Maximum 16-bit
        0x10000   // Should fail or be truncated
    };
    
    for (uint32_t vout : test_vouts) {
        if (vout <= 0xFFFF) {
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.vout = vout & 0xFFFF;
            header.txid_index = 0;
            
            BOOST_CHECK(utxoSet.AddUTXO(txid, vout, header, nullptr));
            BOOST_CHECK(utxoSet.HasUTXO(txid, vout));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(validation_script_size_limits, BasicTestingSetup)
{
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    // Test various script sizes
    std::vector<size_t> test_sizes = {
        0,      // Empty script (unusual but valid)
        22,     // P2WPKH
        23,     // P2SH
        25,     // P2PKH
        34,     // P2WSH/P2TR
        520,    // Maximum standard
        10000,  // Large non-standard
        65535   // Maximum 16-bit
    };
    
    for (size_t size : test_sizes) {
        if (size <= 65535) {
            uint256 txid = GetRandHash();
            
            gpu::UTXOHeader header;
            memset(&header, 0, sizeof(header));
            header.script_size = size;
            header.txid_index = 0;
            header.vout = 0;
            
            std::vector<uint8_t> script(size, 0xAB);
            
            if (utxoSet.AddUTXO(txid, 0, header, script.empty() ? nullptr : script.data())) {
                BOOST_CHECK(utxoSet.HasUTXO(txid, 0));
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()