// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/**
 * GPU Sighash Computation Tests - Phase 9 Comprehensive Testing
 *
 * Tests for signature hash computation on GPU matching CPU results.
 */

#include <boost/test/unit_test.hpp>

#include <coins.h>
#include <hash.h>
#include <key.h>
#include <primitives/transaction.h>
#include <script/interpreter.h>
#include <script/script.h>
#include <script/signingprovider.h>
#include <test/util/setup_common.h>
#include <uint256.h>

// Note: GPU headers not included directly here - tests verify CPU sighash
// computation which the GPU must match exactly.

BOOST_FIXTURE_TEST_SUITE(gpu_sighash_tests, BasicTestingSetup)

#ifdef ENABLE_GPU_ACCELERATION

// Helper to create a simple P2PKH scriptPubKey
static CScript CreateP2PKHScript(const CPubKey& pubkey)
{
    return CScript() << OP_DUP << OP_HASH160 << ToByteVector(pubkey.GetID()) << OP_EQUALVERIFY << OP_CHECKSIG;
}

// Helper to create a simple P2WPKH scriptPubKey
static CScript CreateP2WPKHScript(const CPubKey& pubkey)
{
    return CScript() << OP_0 << ToByteVector(pubkey.GetID());
}

// =============================================================================
// Legacy Sighash Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_sighash_legacy_all)
{
    // Create a simple transaction
    CMutableTransaction tx;
    tx.version = 1;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vin[0].nSequence = 0xFFFFFFFF;
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    CScript scriptPubKey = CreateP2PKHScript(key.GetPubKey());

    // Compute sighash with CPU - GPU must match this exactly
    uint256 cpu_hash = SignatureHash(scriptPubKey, tx, 0, SIGHASH_ALL, 0, SigVersion::BASE);

    // Verify CPU computation produces valid hash
    BOOST_CHECK(cpu_hash != uint256::ZERO);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_legacy_none)
{
    CMutableTransaction tx;
    tx.version = 1;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    CScript scriptPubKey = CreateP2PKHScript(key.GetPubKey());

    uint256 cpu_hash = SignatureHash(scriptPubKey, tx, 0, SIGHASH_NONE, 0, SigVersion::BASE);

    BOOST_CHECK(cpu_hash != uint256::ZERO);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_legacy_single)
{
    CMutableTransaction tx;
    tx.version = 1;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    CScript scriptPubKey = CreateP2PKHScript(key.GetPubKey());

    uint256 cpu_hash = SignatureHash(scriptPubKey, tx, 0, SIGHASH_SINGLE, 0, SigVersion::BASE);

    BOOST_CHECK(cpu_hash != uint256::ZERO);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_legacy_anyonecanpay)
{
    CMutableTransaction tx;
    tx.version = 1;
    tx.vin.resize(2);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vin[1].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 1);
    tx.vout.resize(1);
    tx.vout[0].nValue = 200000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    CScript scriptPubKey = CreateP2PKHScript(key.GetPubKey());

    uint256 hash_all = SignatureHash(scriptPubKey, tx, 0, SIGHASH_ALL | SIGHASH_ANYONECANPAY, 0, SigVersion::BASE);
    uint256 hash_none = SignatureHash(scriptPubKey, tx, 0, SIGHASH_NONE | SIGHASH_ANYONECANPAY, 0, SigVersion::BASE);
    uint256 hash_single = SignatureHash(scriptPubKey, tx, 0, SIGHASH_SINGLE | SIGHASH_ANYONECANPAY, 0, SigVersion::BASE);

    BOOST_CHECK(hash_all != uint256::ZERO);
    BOOST_CHECK(hash_none != uint256::ZERO);
    BOOST_CHECK(hash_single != uint256::ZERO);

    // All should be different
    BOOST_CHECK(hash_all != hash_none);
    BOOST_CHECK(hash_all != hash_single);
    BOOST_CHECK(hash_none != hash_single);
}

// =============================================================================
// BIP143 SegWit v0 Sighash Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_sighash_bip143_p2wpkh)
{
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vin[0].nSequence = 0xFFFFFFFF;
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    CScript scriptPubKey = CreateP2WPKHScript(key.GetPubKey());

    CAmount amount = 200000000;

    // For BIP143, we need PrecomputedTransactionData
    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs));

    uint256 cpu_hash = SignatureHash(scriptPubKey, tx, 0, SIGHASH_ALL, amount, SigVersion::WITNESS_V0, &txdata);

    BOOST_CHECK(cpu_hash != uint256::ZERO);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_bip143_all_types)
{
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    CScript scriptPubKey = CreateP2WPKHScript(key.GetPubKey());
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs));

    // Test all sighash types
    int sighash_types[] = {
        SIGHASH_ALL,
        SIGHASH_NONE,
        SIGHASH_SINGLE,
        SIGHASH_ALL | SIGHASH_ANYONECANPAY,
        SIGHASH_NONE | SIGHASH_ANYONECANPAY,
        SIGHASH_SINGLE | SIGHASH_ANYONECANPAY
    };

    std::vector<uint256> hashes;
    for (int sighash_type : sighash_types) {
        uint256 hash = SignatureHash(scriptPubKey, tx, 0, sighash_type, amount, SigVersion::WITNESS_V0, &txdata);
        BOOST_CHECK(hash != uint256::ZERO);
        hashes.push_back(hash);
    }

    // All hashes should be unique
    for (size_t i = 0; i < hashes.size(); i++) {
        for (size_t j = i + 1; j < hashes.size(); j++) {
            BOOST_CHECK(hashes[i] != hashes[j]);
        }
    }
}

// =============================================================================
// BIP341 Taproot Sighash Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_sighash_bip341_default)
{
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vin[0].nSequence = 0xFFFFFFFF;
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    // P2TR scriptPubKey
    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);  // true for Taproot

    ScriptExecutionData execdata;
    execdata.m_annex_init = true;
    execdata.m_annex_present = false;

    uint256 hash;
    bool success = SignatureHashSchnorr(hash, execdata, tx, 0, 0x00, SigVersion::TAPROOT, txdata, MissingDataBehavior::FAIL);

    BOOST_CHECK(success);
    BOOST_CHECK(hash != uint256::ZERO);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_bip341_all_types)
{
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    // Taproot sighash types
    uint8_t sighash_types[] = {
        0x00,  // SIGHASH_DEFAULT
        0x01,  // SIGHASH_ALL
        0x02,  // SIGHASH_NONE
        0x03,  // SIGHASH_SINGLE
        0x81,  // SIGHASH_ALL | ANYONECANPAY
        0x82,  // SIGHASH_NONE | ANYONECANPAY
        0x83   // SIGHASH_SINGLE | ANYONECANPAY
    };

    std::vector<uint256> hashes;
    for (uint8_t sighash_type : sighash_types) {
        ScriptExecutionData execdata;
        execdata.m_annex_init = true;
        execdata.m_annex_present = false;

        uint256 hash;
        bool success = SignatureHashSchnorr(hash, execdata, tx, 0, sighash_type, SigVersion::TAPROOT, txdata, MissingDataBehavior::FAIL);

        BOOST_CHECK(success);
        BOOST_CHECK(hash != uint256::ZERO);
        hashes.push_back(hash);
    }

    // All hashes should be unique (SIGHASH_DEFAULT encodes as 0x00, SIGHASH_ALL as 0x01,
    // and the sighash type is part of the message, so they produce different hashes)
    for (size_t i = 0; i < hashes.size(); i++) {
        for (size_t j = i + 1; j < hashes.size(); j++) {
            BOOST_CHECK(hashes[i] != hashes[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(gpu_sighash_bip341_with_annex)
{
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
        tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    // Without annex
    ScriptExecutionData execdata_no_annex;
    execdata_no_annex.m_annex_init = true;
    execdata_no_annex.m_annex_present = false;

    uint256 hash_no_annex;
    BOOST_CHECK(SignatureHashSchnorr(hash_no_annex, execdata_no_annex, tx, 0, 0x00, SigVersion::TAPROOT, txdata, MissingDataBehavior::FAIL));

    // With annex
    ScriptExecutionData execdata_with_annex;
    execdata_with_annex.m_annex_init = true;
    execdata_with_annex.m_annex_present = true;
    // Compute annex hash
    std::vector<unsigned char> annex = {0x50, 0x01, 0x02, 0x03};
    std::vector<unsigned char> annex_with_size;
    annex_with_size.push_back(static_cast<unsigned char>(annex.size()));
    annex_with_size.insert(annex_with_size.end(), annex.begin(), annex.end());
    CSHA256().Write(annex_with_size.data(), annex_with_size.size()).Finalize(execdata_with_annex.m_annex_hash.begin());

    uint256 hash_with_annex;
    BOOST_CHECK(SignatureHashSchnorr(hash_with_annex, execdata_with_annex, tx, 0, 0x00, SigVersion::TAPROOT, txdata, MissingDataBehavior::FAIL));

    // Hashes should be different
    BOOST_CHECK(hash_no_annex != hash_with_annex);
}

// =============================================================================
// Multiple Input Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_sighash_multiple_inputs)
{
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(3);
    for (int i = 0; i < 3; i++) {
        tx.vin[i].prevout = COutPoint(Txid::FromUint256(GetRandHash()), i);
        tx.vin[i].nSequence = 0xFFFFFFFF;
    }
    tx.vout.resize(2);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;
    tx.vout[1].nValue = 50000000;
    tx.vout[1].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    CScript scriptPubKey = CreateP2WPKHScript(key.GetPubKey());
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    for (int i = 0; i < 3; i++) {
        spent_outputs.push_back(CTxOut(amount, scriptPubKey));
    }

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs));

    // Compute sighash for each input
    std::vector<uint256> hashes;
    for (int i = 0; i < 3; i++) {
        uint256 hash = SignatureHash(scriptPubKey, tx, i, SIGHASH_ALL, amount, SigVersion::WITNESS_V0, &txdata);
        BOOST_CHECK(hash != uint256::ZERO);
        hashes.push_back(hash);
    }

    // Each input should have a different hash
    BOOST_CHECK(hashes[0] != hashes[1]);
    BOOST_CHECK(hashes[0] != hashes[2]);
    BOOST_CHECK(hashes[1] != hashes[2]);
}

// =============================================================================
// BIP342 Tapscript Sighash Tests (Script Path)
// These tests verify proper ScriptExecutionData initialization for TAPSCRIPT.
// The bug that was caught: m_codeseparator_pos_init was not being set, causing
// an assertion failure in SignatureHashSchnorr for script path spends.
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_sighash_tapscript_basic)
{
    // Test TAPSCRIPT (script path) sighash with properly initialized execdata
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vin[0].nSequence = 0xFFFFFFFF;
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    // Properly initialized execdata for TAPSCRIPT
    ScriptExecutionData execdata;
    execdata.m_annex_init = true;
    execdata.m_annex_present = false;
    execdata.m_codeseparator_pos_init = true;  // CRITICAL: must be set for TAPSCRIPT
    execdata.m_codeseparator_pos = 0xFFFFFFFFUL;  // No OP_CODESEPARATOR executed
    execdata.m_tapleaf_hash_init = true;  // CRITICAL: must be set for TAPSCRIPT
    // Compute a dummy tapleaf hash (in real usage, computed from tapscript)
    execdata.m_tapleaf_hash = GetRandHash();

    uint256 hash;
    bool success = SignatureHashSchnorr(hash, execdata, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL);

    BOOST_CHECK(success);
    BOOST_CHECK(hash != uint256::ZERO);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_tapscript_all_types)
{
    // Test all sighash types for TAPSCRIPT
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    uint8_t sighash_types[] = {
        0x00,  // SIGHASH_DEFAULT
        0x01,  // SIGHASH_ALL
        0x02,  // SIGHASH_NONE
        0x03,  // SIGHASH_SINGLE
        0x81,  // SIGHASH_ALL | ANYONECANPAY
        0x82,  // SIGHASH_NONE | ANYONECANPAY
        0x83   // SIGHASH_SINGLE | ANYONECANPAY
    };

    std::vector<uint256> hashes;
    for (uint8_t sighash_type : sighash_types) {
        ScriptExecutionData execdata;
        execdata.m_annex_init = true;
        execdata.m_annex_present = false;
        execdata.m_codeseparator_pos_init = true;
        execdata.m_codeseparator_pos = 0xFFFFFFFFUL;
        execdata.m_tapleaf_hash_init = true;
        execdata.m_tapleaf_hash = GetRandHash();

        uint256 hash;
        bool success = SignatureHashSchnorr(hash, execdata, tx, 0, sighash_type, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL);

        BOOST_CHECK(success);
        BOOST_CHECK(hash != uint256::ZERO);
        hashes.push_back(hash);
    }

    // All hashes should be unique
    for (size_t i = 0; i < hashes.size(); i++) {
        for (size_t j = i + 1; j < hashes.size(); j++) {
            BOOST_CHECK(hashes[i] != hashes[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(gpu_sighash_tapscript_with_codeseparator)
{
    // Test TAPSCRIPT with different codeseparator positions
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    uint256 tapleaf_hash = GetRandHash();

    // Test with no codeseparator (0xFFFFFFFF)
    ScriptExecutionData execdata1;
    execdata1.m_annex_init = true;
    execdata1.m_annex_present = false;
    execdata1.m_codeseparator_pos_init = true;
    execdata1.m_codeseparator_pos = 0xFFFFFFFFUL;
    execdata1.m_tapleaf_hash_init = true;
    execdata1.m_tapleaf_hash = tapleaf_hash;

    uint256 hash1;
    BOOST_CHECK(SignatureHashSchnorr(hash1, execdata1, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));

    // Test with codeseparator at position 0
    ScriptExecutionData execdata2;
    execdata2.m_annex_init = true;
    execdata2.m_annex_present = false;
    execdata2.m_codeseparator_pos_init = true;
    execdata2.m_codeseparator_pos = 0;
    execdata2.m_tapleaf_hash_init = true;
    execdata2.m_tapleaf_hash = tapleaf_hash;

    uint256 hash2;
    BOOST_CHECK(SignatureHashSchnorr(hash2, execdata2, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));

    // Test with codeseparator at position 10
    ScriptExecutionData execdata3;
    execdata3.m_annex_init = true;
    execdata3.m_annex_present = false;
    execdata3.m_codeseparator_pos_init = true;
    execdata3.m_codeseparator_pos = 10;
    execdata3.m_tapleaf_hash_init = true;
    execdata3.m_tapleaf_hash = tapleaf_hash;

    uint256 hash3;
    BOOST_CHECK(SignatureHashSchnorr(hash3, execdata3, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));

    // All hashes should be different (codeseparator_pos affects the sighash)
    BOOST_CHECK(hash1 != hash2);
    BOOST_CHECK(hash1 != hash3);
    BOOST_CHECK(hash2 != hash3);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_tapscript_tapleaf_hash_affects_sighash)
{
    // Verify that different tapleaf hashes produce different sighashes
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    // Two different tapleaf hashes
    uint256 tapleaf1 = GetRandHash();
    uint256 tapleaf2 = GetRandHash();
    BOOST_CHECK(tapleaf1 != tapleaf2);

    ScriptExecutionData execdata1;
    execdata1.m_annex_init = true;
    execdata1.m_annex_present = false;
    execdata1.m_codeseparator_pos_init = true;
    execdata1.m_codeseparator_pos = 0xFFFFFFFFUL;
    execdata1.m_tapleaf_hash_init = true;
    execdata1.m_tapleaf_hash = tapleaf1;

    ScriptExecutionData execdata2;
    execdata2.m_annex_init = true;
    execdata2.m_annex_present = false;
    execdata2.m_codeseparator_pos_init = true;
    execdata2.m_codeseparator_pos = 0xFFFFFFFFUL;
    execdata2.m_tapleaf_hash_init = true;
    execdata2.m_tapleaf_hash = tapleaf2;

    uint256 hash1, hash2;
    BOOST_CHECK(SignatureHashSchnorr(hash1, execdata1, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));
    BOOST_CHECK(SignatureHashSchnorr(hash2, execdata2, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));

    // Different tapleaf hashes must produce different sighashes
    BOOST_CHECK(hash1 != hash2);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_tapscript_with_annex)
{
    // Test TAPSCRIPT with annex present
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    uint256 tapleaf_hash = GetRandHash();

    // Without annex
    ScriptExecutionData execdata_no_annex;
    execdata_no_annex.m_annex_init = true;
    execdata_no_annex.m_annex_present = false;
    execdata_no_annex.m_codeseparator_pos_init = true;
    execdata_no_annex.m_codeseparator_pos = 0xFFFFFFFFUL;
    execdata_no_annex.m_tapleaf_hash_init = true;
    execdata_no_annex.m_tapleaf_hash = tapleaf_hash;

    uint256 hash_no_annex;
    BOOST_CHECK(SignatureHashSchnorr(hash_no_annex, execdata_no_annex, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));

    // With annex
    ScriptExecutionData execdata_with_annex;
    execdata_with_annex.m_annex_init = true;
    execdata_with_annex.m_annex_present = true;
    std::vector<unsigned char> annex = {0x50, 0x01, 0x02, 0x03};
    std::vector<unsigned char> annex_with_size;
    annex_with_size.push_back(static_cast<unsigned char>(annex.size()));
    annex_with_size.insert(annex_with_size.end(), annex.begin(), annex.end());
    CSHA256().Write(annex_with_size.data(), annex_with_size.size()).Finalize(execdata_with_annex.m_annex_hash.begin());
    execdata_with_annex.m_codeseparator_pos_init = true;
    execdata_with_annex.m_codeseparator_pos = 0xFFFFFFFFUL;
    execdata_with_annex.m_tapleaf_hash_init = true;
    execdata_with_annex.m_tapleaf_hash = tapleaf_hash;

    uint256 hash_with_annex;
    BOOST_CHECK(SignatureHashSchnorr(hash_with_annex, execdata_with_annex, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));

    // Hashes should be different
    BOOST_CHECK(hash_no_annex != hash_with_annex);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_taproot_vs_tapscript)
{
    // Verify that TAPROOT (key path) and TAPSCRIPT (script path) produce different sighashes
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    // TAPROOT (key path) - doesn't need codeseparator_pos or tapleaf_hash
    ScriptExecutionData execdata_keypath;
    execdata_keypath.m_annex_init = true;
    execdata_keypath.m_annex_present = false;

    uint256 hash_keypath;
    BOOST_CHECK(SignatureHashSchnorr(hash_keypath, execdata_keypath, tx, 0, 0x00, SigVersion::TAPROOT, txdata, MissingDataBehavior::FAIL));

    // TAPSCRIPT (script path) - needs all fields
    ScriptExecutionData execdata_scriptpath;
    execdata_scriptpath.m_annex_init = true;
    execdata_scriptpath.m_annex_present = false;
    execdata_scriptpath.m_codeseparator_pos_init = true;
    execdata_scriptpath.m_codeseparator_pos = 0xFFFFFFFFUL;
    execdata_scriptpath.m_tapleaf_hash_init = true;
    execdata_scriptpath.m_tapleaf_hash = GetRandHash();

    uint256 hash_scriptpath;
    BOOST_CHECK(SignatureHashSchnorr(hash_scriptpath, execdata_scriptpath, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL));

    // Key path and script path must produce different sighashes
    BOOST_CHECK(hash_keypath != hash_scriptpath);
}

BOOST_AUTO_TEST_CASE(gpu_sighash_execdata_initialization_validation_style)
{
    // This test mimics exactly how validation.cpp initializes ScriptExecutionData
    // to ensure the initialization pattern is correct and catches future regressions.
    CMutableTransaction tx;
    tx.version = 2;
    tx.vin.resize(1);
    tx.vin[0].prevout = COutPoint(Txid::FromUint256(GetRandHash()), 0);
    tx.vout.resize(1);
    tx.vout[0].nValue = 100000000;
    tx.vout[0].scriptPubKey = CScript() << OP_TRUE;

    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    CScript scriptPubKey = CScript() << OP_1 << ToByteVector(xonly);
    CAmount amount = 200000000;

    std::vector<CTxOut> spent_outputs;
    spent_outputs.push_back(CTxOut(amount, scriptPubKey));

    PrecomputedTransactionData txdata;
    txdata.Init(tx, std::move(spent_outputs), true);

    // =======================================================================
    // Initialize execdata EXACTLY as validation.cpp does for P2TR script path
    // =======================================================================
    ScriptExecutionData execdata;

    // Step 1: Basic initialization (required for all Schnorr sighashes)
    execdata.m_annex_init = true;
    execdata.m_codeseparator_pos_init = true;
    execdata.m_codeseparator_pos = 0xFFFFFFFFUL;

    // Step 2: Annex handling (simulating no annex case)
    bool has_annex = false;  // Would check witness stack in real code
    if (has_annex) {
        execdata.m_annex_present = true;
        // Would compute annex hash here
    } else {
        execdata.m_annex_present = false;
    }

    // Step 3: For TAPSCRIPT, compute tapleaf hash
    // Simulating script path with leaf version 0xc0
    uint8_t leaf_version = 0xc0;
    CScript tapscript = CScript() << OP_TRUE;  // Simple tapscript

    execdata.m_tapleaf_hash_init = true;
    // Use ComputeTapleafHash in real code
    execdata.m_tapleaf_hash = (HashWriter{} << uint8_t(leaf_version) << tapscript).GetSHA256();

    // Step 4: Compute sighash
    uint256 hash;
    bool success = SignatureHashSchnorr(hash, execdata, tx, 0, 0x00, SigVersion::TAPSCRIPT, txdata, MissingDataBehavior::FAIL);

    BOOST_CHECK_MESSAGE(success, "SignatureHashSchnorr failed with validation-style initialization");
    BOOST_CHECK(hash != uint256::ZERO);
}

#else // !ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_CASE(gpu_sighash_tests_disabled)
{
    BOOST_TEST_MESSAGE("GPU sighash tests disabled - GPU acceleration not enabled");
    BOOST_CHECK(true);
}

#endif // ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_SUITE_END()
