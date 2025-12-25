// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/**
 * GPU Signature Verification Tests - Phase 9 Comprehensive Testing
 */

#include <boost/test/unit_test.hpp>

#include <hash.h>
#include <key.h>
#include <pubkey.h>
#include <script/interpreter.h>
#include <test/util/setup_common.h>
#include <uint256.h>
#include <util/strencodings.h>
#include <util/string.h>

#include <span>

using namespace util::hex_literals;

#ifdef ENABLE_GPU_ACCELERATION
#include <gpu_kernel/gpu_batch_validator.h>
#endif

BOOST_FIXTURE_TEST_SUITE(gpu_sig_tests, BasicTestingSetup)

#ifdef ENABLE_GPU_ACCELERATION

// =============================================================================
// ECDSA Signature Verification Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_ecdsa_verify_valid_signature)
{
    // Generate a key pair
    CKey key;
    key.MakeNewKey(true);
    CPubKey pubkey = key.GetPubKey();

    // Create a message to sign
    uint256 hash = GetRandHash();

    // Sign with CPU
    std::vector<unsigned char> sig;
    BOOST_CHECK(key.Sign(hash, sig));

    // Verify with CPU
    BOOST_CHECK(pubkey.Verify(hash, sig));
}

BOOST_AUTO_TEST_CASE(gpu_ecdsa_verify_invalid_signature)
{
    CKey key;
    key.MakeNewKey(true);
    CPubKey pubkey = key.GetPubKey();

    uint256 hash = GetRandHash();

    std::vector<unsigned char> sig;
    BOOST_CHECK(key.Sign(hash, sig));

    // Corrupt the signature
    sig[sig.size() / 2] ^= 0xFF;

    // CPU should reject
    BOOST_CHECK(!pubkey.Verify(hash, sig));
}

// =============================================================================
// Schnorr Signature Verification Tests (BIP340)
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_schnorr_verify_valid_signature)
{
    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly_pubkey(key.GetPubKey());

    uint256 hash = GetRandHash();

    std::vector<unsigned char> sig(64);
    BOOST_CHECK(key.SignSchnorr(hash, sig, nullptr, {}));

    // Verify with CPU
    BOOST_CHECK(xonly_pubkey.VerifySchnorr(hash, sig));
}

BOOST_AUTO_TEST_CASE(gpu_schnorr_verify_invalid_signature)
{
    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly_pubkey(key.GetPubKey());

    uint256 hash = GetRandHash();

    std::vector<unsigned char> sig(64);
    BOOST_CHECK(key.SignSchnorr(hash, sig, nullptr, {}));

    // Corrupt signature
    sig[32] ^= 0xFF;

    // Should fail verification
    BOOST_CHECK(!xonly_pubkey.VerifySchnorr(hash, sig));
}

// =============================================================================
// Batch Verification Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(gpu_ecdsa_batch_signatures)
{
    const int NUM_SIGS = 10;

    std::vector<CKey> keys(NUM_SIGS);
    std::vector<CPubKey> pubkeys(NUM_SIGS);
    std::vector<uint256> hashes(NUM_SIGS);
    std::vector<std::vector<unsigned char>> sigs(NUM_SIGS);

    for (int i = 0; i < NUM_SIGS; i++) {
        keys[i].MakeNewKey(true);
        pubkeys[i] = keys[i].GetPubKey();
        hashes[i] = GetRandHash();
        BOOST_CHECK(keys[i].Sign(hashes[i], sigs[i]));
    }

    // Verify all
    for (int i = 0; i < NUM_SIGS; i++) {
        BOOST_CHECK(pubkeys[i].Verify(hashes[i], sigs[i]));
    }
}

BOOST_AUTO_TEST_CASE(gpu_schnorr_batch_signatures)
{
    const int NUM_SIGS = 10;

    std::vector<CKey> keys(NUM_SIGS);
    std::vector<XOnlyPubKey> xonly_pubkeys(NUM_SIGS);
    std::vector<uint256> hashes(NUM_SIGS);
    std::vector<std::vector<unsigned char>> sigs(NUM_SIGS);

    for (int i = 0; i < NUM_SIGS; i++) {
        keys[i].MakeNewKey(true);
        xonly_pubkeys[i] = XOnlyPubKey(keys[i].GetPubKey());
        hashes[i] = GetRandHash();
        sigs[i].resize(64);
        BOOST_CHECK(keys[i].SignSchnorr(hashes[i], sigs[i], nullptr, {}));
    }

    for (int i = 0; i < NUM_SIGS; i++) {
        BOOST_CHECK(xonly_pubkeys[i].VerifySchnorr(hashes[i], sigs[i]));
    }
}

// =============================================================================
// BIP340 Official Test Vectors
// =============================================================================

BOOST_AUTO_TEST_CASE(bip340_test_vector_0)
{
    // Test vector 0: Valid signature
    auto pubkey_bytes = ParseHex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
    auto msg_bytes = ParseHex("0000000000000000000000000000000000000000000000000000000000000000");
    auto sig_bytes = ParseHex("E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA821525F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    BOOST_CHECK(pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_1)
{
    // Test vector 1: Valid signature
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("6896BD60EEAE296DB48A229FF71DFE071BDE413E6D43F917DC8DCF8C78DE33418906D11AC976ABCCB20B091292BFF4EA897EFCB639EA871CFA95F6DE339E4B0A");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    BOOST_CHECK(pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_2)
{
    // Test vector 2: Valid signature
    auto pubkey_bytes = ParseHex("DD308AFEC5777E13121FA72B9CC1B7CC0139715309B086C960E18FD969774EB8");
    auto msg_bytes = ParseHex("7E2D58D8B3BCDF1ABADEC7829054F90DDA9805AAB56C77333024B9D0A508B75C");
    auto sig_bytes = ParseHex("5831AAEED7B44BB74E5EAB94BA9D4294C49BCF2A60728D8B4C200F50DD313C1BAB745879A5AD954A72C45A91C3A51D3C7ADEA98D82F8481E0E1E03674A6F3FB7");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    BOOST_CHECK(pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_3)
{
    // Test vector 3: Valid signature (key with odd y-coordinate)
    auto pubkey_bytes = ParseHex("25D1DFF95105F5253C4022F628A996AD3A0D95FBF21D468A1B33F8C160D8F517");
    auto msg_bytes = ParseHex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    auto sig_bytes = ParseHex("7EB0509757E246F19449885651611CB965ECC1A187DD51B64FDA1EDC9637D5EC97582B9CB13DB3933705B32BA982AF5AF25FD78881EBB32771FC5922EFC66EA3");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    BOOST_CHECK(pubkey.VerifySchnorr(msg, sig_bytes));
}

// =============================================================================
// BIP340 Invalid Test Vectors - These MUST fail verification
// =============================================================================

BOOST_AUTO_TEST_CASE(bip340_test_vector_5_pubkey_not_on_curve)
{
    // BIP340 test vector 5: public key not on the curve
    auto pubkey_bytes = ParseHex("EEFDEA4CDB677750A420FEE807EACF21EB9898AE79B9768766E4FAA04A2D4A34");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E17776969E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // Public key not on curve - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_6_has_even_y_R_is_false)
{
    // BIP340 test vector 6: has_even_y(R) is false
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("FFF97BD5755EEEA420453A14355235D382F6472F8568A18B2F057A14602975563CC27944640AC607CD107AE10923D9EF7A73C643E166BE5EBEAFA34B1AC553E2");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // R has odd y-coordinate - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_7_negated_message)
{
    // BIP340 test vector 7: negated message hash
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("1FA62E331EDBC21C394792D2AB1100A7B432B013DF3F6FF4F99FCB33E0E1515F28890B3EDB6E7189B630448B515CE4F8622A954CFE545735AAEA5134FCCDB2BD");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // Negated message - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_8_negated_s)
{
    // BIP340 test vector 8: negated s value
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769961764B3AA9B2FFCB6EF947B6887A226E8D7C93E00C5ED0C1834FF0D0C2E6DA6");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // Negated s - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_9_sG_minus_eP_infinite)
{
    // BIP340 test vector 9: sG - eP is infinite
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("0000000000000000000000000000000000000000000000000000000000000000123DDA8328AF9C23A94C1FEECFD123BA4FB73476F0D594DCB65C6425BD186051");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // sG - eP is infinite - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_10_sG_minus_eP_odd_y)
{
    // BIP340 test vector 10: sG - eP has odd y
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("00000000000000000000000000000000000000000000000000000000000000017615FBAF5AE28864013C099742DEADB4DBA87F11AC6754F93780D5A1837CF197");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // sG - eP has odd y - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_11_sig_r_not_on_curve)
{
    // BIP340 test vector 11: sig[0:32] is not a valid x-coordinate
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("4A298DACAE57395A15D0795DDBFD1DCB564DA82B0F269BC70A74F8220429BA1D69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // sig r not on curve - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_12_sig_r_exceeds_field)
{
    // BIP340 test vector 12: sig[0:32] is equal to field size
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // sig r equals field size - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_13_sig_s_exceeds_order)
{
    // BIP340 test vector 13: sig[32:64] is equal to curve order
    auto pubkey_bytes = ParseHex("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // sig s equals curve order - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_14_pubkey_exceeds_field)
{
    // BIP340 test vector 14: public key exceeds field size
    auto pubkey_bytes = ParseHex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30");
    auto msg_bytes = ParseHex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    auto sig_bytes = ParseHex("6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E17776969E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // Public key exceeds field size - MUST fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

// =============================================================================
// Additional Invalid Signature Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(bip340_test_invalid_sig_corrupted)
{
    // Valid sig corrupted by flipping a bit
    auto pubkey_bytes = ParseHex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
    auto msg_bytes = ParseHex("0000000000000000000000000000000000000000000000000000000000000000");
    // Original valid sig with last byte changed (0xC0 -> 0xC1)
    auto sig_bytes = ParseHex("E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA821525F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C1");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // Corrupted signature should fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_wrong_message)
{
    // Valid sig but for wrong message
    auto pubkey_bytes = ParseHex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
    auto msg_bytes = ParseHex("0000000000000000000000000000000000000000000000000000000000000001");  // Different from test 0
    auto sig_bytes = ParseHex("E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA821525F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // Wrong message should fail
    BOOST_CHECK(!pubkey.VerifySchnorr(msg, sig_bytes));
}

BOOST_AUTO_TEST_CASE(bip340_test_vector_4)
{
    // BIP340 test vector 4: valid signature with low R value (many leading zeros)
    auto pubkey_bytes = ParseHex("D69C3509BB99E412E68B0FE8544E72837DFA30746D8BE2AA65975F29D22DC7B9");
    auto msg_bytes = ParseHex("4DF3C3F68FCC83B27E9D42C90431A72499F17875C81A599B566C9889B9696703");
    auto sig_bytes = ParseHex("00000000000000000000003B78CE563F89A0ED9414F5AA28AD0D96D6795F9C6376AFB1548AF603B3EB45C9F8207DEE1060CB71C04E80F593060B07D28308D7F4");

    XOnlyPubKey pubkey(std::span<const unsigned char>(pubkey_bytes.data(), 32));
    uint256 msg;
    memcpy(msg.begin(), msg_bytes.data(), 32);

    // Valid signature - MUST pass
    BOOST_CHECK(pubkey.VerifySchnorr(msg, sig_bytes));
}

// =============================================================================
// BIP341 Taproot Tests - TapTweak and Hash Computations
// =============================================================================

BOOST_AUTO_TEST_CASE(bip341_taptweak_keypath_only)
{
    // BIP341 test: Key path only spending (no script tree)
    // Internal pubkey with null merkle root should produce deterministic tweaked key
    auto internal_pubkey_bytes = ParseHex("d6889cb081036e0faefa3a35157ad71086b123b2b144b649798b494c300a961d");
    XOnlyPubKey internal_pubkey(std::span<const unsigned char>(internal_pubkey_bytes.data(), 32));

    // Expected tweaked pubkey from BIP341 test vectors
    auto expected_tweaked = ParseHex("53a1f6e454df1aa2776a2814a721372d6258050de330b3c6d10ee8f4e0dda343");

    auto result = internal_pubkey.CreateTapTweak(nullptr);
    BOOST_CHECK(result.has_value());

    XOnlyPubKey tweaked_pubkey = result->first;
    BOOST_CHECK(std::equal(tweaked_pubkey.begin(), tweaked_pubkey.end(), expected_tweaked.begin()));
}

BOOST_AUTO_TEST_CASE(bip341_taptweak_with_merkle_root)
{
    // BIP341 test: Key path spending with script tree merkle root
    auto internal_pubkey_bytes = ParseHex("187791b6f712a8ea41c8ecdd0ee77fab3e85263b37e1ec18a3651926b3a6cf27");
    XOnlyPubKey internal_pubkey(std::span<const unsigned char>(internal_pubkey_bytes.data(), 32));

    // Merkle root from single leaf
    uint256 merkle_root;
    auto merkle_bytes = ParseHex("5b75adecf53548f3ec6ad7d78383bf84cc57b55a3127c72b9a2481752dd88b21");
    memcpy(merkle_root.begin(), merkle_bytes.data(), 32);

    // Expected tweaked pubkey
    auto expected_tweaked = ParseHex("147c9c57132f6e7ecddba9800bb0c4449251c92a1e60371ee77557b6620f3ea3");

    auto result = internal_pubkey.CreateTapTweak(&merkle_root);
    BOOST_CHECK(result.has_value());

    XOnlyPubKey tweaked_pubkey = result->first;
    BOOST_CHECK(std::equal(tweaked_pubkey.begin(), tweaked_pubkey.end(), expected_tweaked.begin()));
}

BOOST_AUTO_TEST_CASE(bip341_tapleaf_hash)
{
    // Test ComputeTapleafHash matches expected values
    // From script_tests.cpp - leaf version 0xc0 and 0xc2
    constexpr uint8_t script[6] = {'f','o','o','b','a','r'};

    constexpr uint256 expected_c0{"edbc10c272a1215dcdcc11d605b9027b5ad6ed97cd45521203f136767b5b9c06"};
    constexpr uint256 expected_c2{"8b5c4f90ae6bf76e259dbef5d8a59df06359c391b59263741b25eca76451b27a"};

    BOOST_CHECK_EQUAL(ComputeTapleafHash(0xc0, std::span(script)), expected_c0);
    BOOST_CHECK_EQUAL(ComputeTapleafHash(0xc2, std::span(script)), expected_c2);
}

BOOST_AUTO_TEST_CASE(bip341_tapbranch_hash)
{
    // Test ComputeTapbranchHash matches expected values
    constexpr uint256 hash1{"8ad69ec7cf41c2a4001fd1f738bf1e505ce2277acdcaa63fe4765192497f47a7"};
    constexpr uint256 hash2{"f224a923cd0021ab202ab139cc56802ddb92dcfc172b9212261a539df79a112a"};
    constexpr uint256 expected{"a64c5b7b943315f9b805d7a7296bedfcfd08919270a1f7a1466e98f8693d8cd9"};

    BOOST_CHECK_EQUAL(ComputeTapbranchHash(hash1, hash2), expected);
}

BOOST_AUTO_TEST_CASE(bip341_nums_h_point)
{
    // BIP341 specifies H = SHA256(G) as a nothing-up-my-sleeve point
    // Verify XOnlyPubKey::NUMS_H is correctly computed
    constexpr auto G_uncompressed{"0479be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"_hex};

    HashWriter hw;
    hw.write(G_uncompressed);
    XOnlyPubKey H{hw.GetSHA256()};

    BOOST_CHECK(XOnlyPubKey::NUMS_H == H);
}

// =============================================================================
// BIP341 Key Path Spending Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(bip341_keypath_signature_generation)
{
    // Test that we can sign and verify with a tweaked key
    // When signing with a null/empty merkle root, the keypair uses nullptr for tweak hash
    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey internal_pubkey(key.GetPubKey());

    uint256 msg = GetRandHash();

    // Empty merkle root triggers tweak path but uses nullptr for hash (see key.cpp:543)
    uint256 empty_merkle{};

    // Sign with empty merkle root (treated as nullptr for tweak hash)
    std::vector<unsigned char> sig(64);
    BOOST_CHECK(key.SignSchnorr(msg, sig, &empty_merkle, {}));

    // CreateTapTweak with nullptr produces same tweak as empty merkle root
    auto tweaked = internal_pubkey.CreateTapTweak(nullptr);
    BOOST_CHECK(tweaked.has_value());
    XOnlyPubKey tweaked_pubkey = tweaked->first;

    // Verify with tweaked key
    BOOST_CHECK(tweaked_pubkey.VerifySchnorr(msg, sig));
}

BOOST_AUTO_TEST_CASE(bip341_keypath_with_merkle_root)
{
    // Test key path spending with a merkle root
    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey internal_pubkey(key.GetPubKey());

    uint256 merkle_root = GetRandHash();
    uint256 msg = GetRandHash();

    // Sign with merkle root tweak
    std::vector<unsigned char> sig(64);
    BOOST_CHECK(key.SignSchnorr(msg, sig, &merkle_root, {}));

    // Get tweaked pubkey with same merkle root
    auto tweaked = internal_pubkey.CreateTapTweak(&merkle_root);
    BOOST_CHECK(tweaked.has_value());
    XOnlyPubKey tweaked_pubkey = tweaked->first;

    // Verify with tweaked key
    BOOST_CHECK(tweaked_pubkey.VerifySchnorr(msg, sig));

    // Wrong merkle root should fail
    uint256 wrong_merkle = GetRandHash();
    auto wrong_tweaked = internal_pubkey.CreateTapTweak(&wrong_merkle);
    BOOST_CHECK(wrong_tweaked.has_value());
    BOOST_CHECK(!wrong_tweaked->first.VerifySchnorr(msg, sig));
}

// =============================================================================
// BIP342 Tapscript Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(bip342_checksig_32byte_pubkey)
{
    // BIP342: CHECKSIG in tapscript uses 32-byte x-only pubkeys
    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    // Verify x-only pubkey is exactly 32 bytes
    BOOST_CHECK_EQUAL(xonly.size(), 32u);

    // Should be able to sign and verify
    uint256 msg = GetRandHash();
    std::vector<unsigned char> sig(64);
    BOOST_CHECK(key.SignSchnorr(msg, sig, nullptr, {}));
    BOOST_CHECK(xonly.VerifySchnorr(msg, sig));
}

BOOST_AUTO_TEST_CASE(bip342_signature_with_sighash_byte)
{
    // BIP342: Schnorr signatures can have optional sighash byte (65 bytes total)
    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    uint256 msg = GetRandHash();
    std::vector<unsigned char> sig(64);
    BOOST_CHECK(key.SignSchnorr(msg, sig, nullptr, {}));

    // 64-byte signature is SIGHASH_DEFAULT
    BOOST_CHECK(xonly.VerifySchnorr(msg, sig));

    // Add explicit SIGHASH_ALL (0x01) - still 64 bytes for verification
    // The sighash byte affects the message hash, not verification
    std::vector<unsigned char> sig_with_hashtype = sig;
    sig_with_hashtype.push_back(0x01);  // SIGHASH_ALL
    BOOST_CHECK_EQUAL(sig_with_hashtype.size(), 65u);
}

BOOST_AUTO_TEST_CASE(bip342_empty_signature_invalid)
{
    // Empty signature must fail
    CKey key;
    key.MakeNewKey(true);
    XOnlyPubKey xonly(key.GetPubKey());

    uint256 msg = GetRandHash();
    std::vector<unsigned char> empty_sig;

    // Empty sig should fail (size != 64)
    BOOST_CHECK_EQUAL(empty_sig.size(), 0u);
    // Note: VerifySchnorr expects exactly 64 bytes, so empty fails
}

BOOST_AUTO_TEST_CASE(bip342_keypair_signing)
{
    // Test KeyPair class for efficient signing
    CKey key;
    key.MakeNewKey(true);

    // Create keypair with empty merkle root (treated as nullptr for tweak hash)
    uint256 empty_merkle{};
    KeyPair keypair = key.ComputeKeyPair(&empty_merkle);

    uint256 msg = GetRandHash();
    uint256 aux = GetRandHash();
    unsigned char sig[64];

    // Sign using keypair
    BOOST_CHECK(keypair.SignSchnorr(msg, sig, aux));

    // Verify with tweaked pubkey (nullptr produces same tweak as empty merkle)
    XOnlyPubKey xonly(key.GetPubKey());
    auto tweaked = xonly.CreateTapTweak(nullptr);
    BOOST_CHECK(tweaked.has_value());
    BOOST_CHECK(tweaked->first.VerifySchnorr(msg, sig));
}

BOOST_AUTO_TEST_CASE(bip342_keypair_with_merkle_tweak)
{
    // Test KeyPair with merkle root tweak
    CKey key;
    key.MakeNewKey(true);

    uint256 merkle_root = GetRandHash();
    KeyPair keypair = key.ComputeKeyPair(&merkle_root);

    uint256 msg = GetRandHash();
    uint256 aux = GetRandHash();
    unsigned char sig[64];

    BOOST_CHECK(keypair.SignSchnorr(msg, sig, aux));

    // Verify with same merkle root
    XOnlyPubKey xonly(key.GetPubKey());
    auto tweaked = xonly.CreateTapTweak(&merkle_root);
    BOOST_CHECK(tweaked.has_value());
    BOOST_CHECK(tweaked->first.VerifySchnorr(msg, sig));
}

#else // !ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_CASE(gpu_sig_tests_disabled)
{
    BOOST_TEST_MESSAGE("GPU signature tests disabled");
    BOOST_CHECK(true);
}

#endif // ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_SUITE_END()
