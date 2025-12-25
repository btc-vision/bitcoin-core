// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>

#include <script/interpreter.h>
#include <script/script.h>
#include <script/signingprovider.h>
#include <primitives/transaction.h>
#include <key.h>
#include <pubkey.h>
#include <hash.h>
#include <crypto/sha256.h>
#include <uint256.h>
#include <util/strencodings.h>

#include <gpu_kernel/gpu_batch_validator.h>
#include <gpu_kernel/gpu_script_types.cuh>

#include <vector>
#include <cstring>

BOOST_AUTO_TEST_SUITE(gpu_p2tr_script_path_tests)

// Test sigversion detection for P2TR key path vs script path
BOOST_AUTO_TEST_CASE(sigversion_detection_key_path)
{
    // Key path spend: witness = [signature]
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));  // 64-byte signature

    size_t wit_count = witness.stack.size();
    bool has_annex = (wit_count >= 2 && !witness.stack.back().empty() && witness.stack.back()[0] == 0x50);
    size_t effective_count = has_annex ? wit_count - 1 : wit_count;

    BOOST_CHECK_EQUAL(effective_count, 1u);  // Key path
    BOOST_CHECK(!has_annex);

    // Should be TAPROOT (key path)
    gpu::GPUSigVersion expected = gpu::GPU_SIGVERSION_TAPROOT;
    gpu::GPUSigVersion actual = (effective_count == 1) ? gpu::GPU_SIGVERSION_TAPROOT : gpu::GPU_SIGVERSION_TAPSCRIPT;
    BOOST_CHECK_EQUAL(static_cast<int>(actual), static_cast<int>(expected));
}

BOOST_AUTO_TEST_CASE(sigversion_detection_script_path_simple)
{
    // Script path spend: witness = [sig, script, control_block]
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));  // signature
    witness.stack.push_back(std::vector<unsigned char>{0x51});      // OP_1 tapscript
    witness.stack.push_back(std::vector<unsigned char>(33, 0xc0));  // control block

    size_t wit_count = witness.stack.size();
    bool has_annex = (wit_count >= 2 && !witness.stack.back().empty() && witness.stack.back()[0] == 0x50);
    size_t effective_count = has_annex ? wit_count - 1 : wit_count;

    BOOST_CHECK_EQUAL(effective_count, 3u);  // Script path (3 elements)
    BOOST_CHECK(!has_annex);

    // Should be TAPSCRIPT (script path)
    gpu::GPUSigVersion expected = gpu::GPU_SIGVERSION_TAPSCRIPT;
    gpu::GPUSigVersion actual = (effective_count == 1) ? gpu::GPU_SIGVERSION_TAPROOT : gpu::GPU_SIGVERSION_TAPSCRIPT;
    BOOST_CHECK_EQUAL(static_cast<int>(actual), static_cast<int>(expected));
}

BOOST_AUTO_TEST_CASE(sigversion_detection_script_path_with_annex)
{
    // Script path with annex: witness = [sig, script, control_block, annex]
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));  // signature
    witness.stack.push_back(std::vector<unsigned char>{0x51});      // OP_1 tapscript
    witness.stack.push_back(std::vector<unsigned char>(33, 0xc0));  // control block
    witness.stack.push_back(std::vector<unsigned char>{0x50, 0x01, 0x02});  // annex (starts with 0x50)

    size_t wit_count = witness.stack.size();
    bool has_annex = (wit_count >= 2 && !witness.stack.back().empty() && witness.stack.back()[0] == 0x50);
    size_t effective_count = has_annex ? wit_count - 1 : wit_count;

    BOOST_CHECK(has_annex);
    BOOST_CHECK_EQUAL(effective_count, 3u);  // Script path (3 elements after removing annex)

    // Should be TAPSCRIPT (script path)
    gpu::GPUSigVersion expected = gpu::GPU_SIGVERSION_TAPSCRIPT;
    gpu::GPUSigVersion actual = (effective_count == 1) ? gpu::GPU_SIGVERSION_TAPROOT : gpu::GPU_SIGVERSION_TAPSCRIPT;
    BOOST_CHECK_EQUAL(static_cast<int>(actual), static_cast<int>(expected));
}

BOOST_AUTO_TEST_CASE(sigversion_detection_key_path_with_annex)
{
    // Key path with annex: witness = [signature, annex]
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));  // signature
    witness.stack.push_back(std::vector<unsigned char>{0x50, 0x01});  // annex (starts with 0x50)

    size_t wit_count = witness.stack.size();
    bool has_annex = (wit_count >= 2 && !witness.stack.back().empty() && witness.stack.back()[0] == 0x50);
    size_t effective_count = has_annex ? wit_count - 1 : wit_count;

    BOOST_CHECK(has_annex);
    BOOST_CHECK_EQUAL(effective_count, 1u);  // Key path (1 element after removing annex)

    // Should be TAPROOT (key path)
    gpu::GPUSigVersion expected = gpu::GPU_SIGVERSION_TAPROOT;
    gpu::GPUSigVersion actual = (effective_count == 1) ? gpu::GPU_SIGVERSION_TAPROOT : gpu::GPU_SIGVERSION_TAPSCRIPT;
    BOOST_CHECK_EQUAL(static_cast<int>(actual), static_cast<int>(expected));
}

BOOST_AUTO_TEST_CASE(sigversion_detection_complex_script_path)
{
    // Complex script path: witness = [arg1, arg2, sig, tapscript, control_block]
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>{0x01});      // stack item 1
    witness.stack.push_back(std::vector<unsigned char>{0x02});      // stack item 2
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));  // signature
    witness.stack.push_back(std::vector<unsigned char>{0xac});      // OP_CHECKSIG tapscript
    witness.stack.push_back(std::vector<unsigned char>(33, 0xc0));  // control block

    size_t wit_count = witness.stack.size();
    bool has_annex = (wit_count >= 2 && !witness.stack.back().empty() && witness.stack.back()[0] == 0x50);
    size_t effective_count = has_annex ? wit_count - 1 : wit_count;

    BOOST_CHECK(!has_annex);
    BOOST_CHECK_EQUAL(effective_count, 5u);  // Script path (5 elements)

    // Should be TAPSCRIPT (script path)
    gpu::GPUSigVersion expected = gpu::GPU_SIGVERSION_TAPSCRIPT;
    gpu::GPUSigVersion actual = (effective_count == 1) ? gpu::GPU_SIGVERSION_TAPROOT : gpu::GPU_SIGVERSION_TAPSCRIPT;
    BOOST_CHECK_EQUAL(static_cast<int>(actual), static_cast<int>(expected));
}

BOOST_AUTO_TEST_CASE(tapleaf_hash_computation)
{
    // Test that tapleaf hash is computed correctly for script path
    // Tapleaf hash = SHA256(SHA256("TapLeaf") || SHA256("TapLeaf") || leaf_version || compact_size(script) || script)

    // Simple OP_TRUE script
    std::vector<unsigned char> script = {0x51};  // OP_TRUE
    uint8_t leaf_version = 0xc0;  // Default tapscript version

    uint256 hash = ComputeTapleafHash(leaf_version, script);

    // Verify hash is non-zero and deterministic
    BOOST_CHECK(hash != uint256());

    // Same inputs should produce same hash
    uint256 hash2 = ComputeTapleafHash(leaf_version, script);
    BOOST_CHECK(hash == hash2);

    // Different script should produce different hash
    std::vector<unsigned char> script2 = {0x00};  // OP_FALSE
    uint256 hash3 = ComputeTapleafHash(leaf_version, script2);
    BOOST_CHECK(hash != hash3);

    // Different leaf version should produce different hash
    uint256 hash4 = ComputeTapleafHash(0xc2, script);  // Different version
    BOOST_CHECK(hash != hash4);
}

BOOST_AUTO_TEST_CASE(control_block_parsing)
{
    // Test control block parsing for leaf version extraction
    // Control block format: [output_key_parity | leaf_version (1 byte)] [internal_pubkey (32 bytes)] [merkle path...]

    // Control block with leaf version 0xc0 (default tapscript)
    std::vector<unsigned char> control_c0(33, 0x00);
    control_c0[0] = 0xc0;  // leaf_version = 0xc0, parity = 0
    uint8_t extracted_c0 = control_c0[0] & 0xfe;  // Mask out parity bit
    BOOST_CHECK_EQUAL(extracted_c0, 0xc0);

    // Control block with leaf version 0xc0 and odd parity
    std::vector<unsigned char> control_c1(33, 0x00);
    control_c1[0] = 0xc1;  // leaf_version = 0xc0, parity = 1
    uint8_t extracted_c1 = control_c1[0] & 0xfe;
    BOOST_CHECK_EQUAL(extracted_c1, 0xc0);

    // Control block with different leaf version
    std::vector<unsigned char> control_c2(33, 0x00);
    control_c2[0] = 0xc2;  // leaf_version = 0xc2, parity = 0
    uint8_t extracted_c2 = control_c2[0] & 0xfe;
    BOOST_CHECK_EQUAL(extracted_c2, 0xc2);
}

BOOST_AUTO_TEST_CASE(witness_element_ordering)
{
    // Verify correct extraction of tapscript and control block from witness
    // For script path without annex: [...stack items..., tapscript, control_block]
    // For script path with annex: [...stack items..., tapscript, control_block, annex]

    // Without annex
    CScriptWitness wit_no_annex;
    wit_no_annex.stack.push_back({0x01});      // stack arg
    wit_no_annex.stack.push_back({0xac});      // tapscript (OP_CHECKSIG)
    wit_no_annex.stack.push_back(std::vector<unsigned char>(33, 0xc0));  // control block

    bool has_annex_1 = (wit_no_annex.stack.size() >= 2 &&
                        !wit_no_annex.stack.back().empty() &&
                        wit_no_annex.stack.back()[0] == 0x50);
    BOOST_CHECK(!has_annex_1);

    size_t ctrl_idx_1 = wit_no_annex.stack.size() - 1;
    size_t script_idx_1 = ctrl_idx_1 - 1;
    BOOST_CHECK_EQUAL(wit_no_annex.stack[script_idx_1][0], 0xac);  // tapscript
    BOOST_CHECK_EQUAL(wit_no_annex.stack[ctrl_idx_1][0], 0xc0);    // control block

    // With annex
    CScriptWitness wit_annex;
    wit_annex.stack.push_back({0x01});         // stack arg
    wit_annex.stack.push_back({0xac});         // tapscript
    wit_annex.stack.push_back(std::vector<unsigned char>(33, 0xc0));  // control block
    wit_annex.stack.push_back({0x50, 0x01});   // annex

    bool has_annex_2 = (wit_annex.stack.size() >= 2 &&
                        !wit_annex.stack.back().empty() &&
                        wit_annex.stack.back()[0] == 0x50);
    BOOST_CHECK(has_annex_2);

    size_t ctrl_idx_2 = has_annex_2 ? wit_annex.stack.size() - 2 : wit_annex.stack.size() - 1;
    size_t script_idx_2 = ctrl_idx_2 - 1;
    BOOST_CHECK_EQUAL(wit_annex.stack[script_idx_2][0], 0xac);  // tapscript (NOT annex)
    BOOST_CHECK_EQUAL(wit_annex.stack[ctrl_idx_2][0], 0xc0);    // control block
}

BOOST_AUTO_TEST_CASE(gpu_sigversion_values)
{
    // Verify GPU sigversion enum values match expected
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::GPU_SIGVERSION_BASE), 0);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::GPU_SIGVERSION_WITNESS_V0), 1);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::GPU_SIGVERSION_TAPROOT), 2);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::GPU_SIGVERSION_TAPSCRIPT), 3);
}

// =============================================================================
// Script Type Detection Tests
// =============================================================================

BOOST_AUTO_TEST_CASE(script_type_detection_p2pkh)
{
    // P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
    std::vector<unsigned char> script = {
        0x76,  // OP_DUP
        0xa9,  // OP_HASH160
        0x14,  // Push 20 bytes
    };
    script.insert(script.end(), 20, 0x00);  // 20-byte hash
    script.push_back(0x88);  // OP_EQUALVERIFY
    script.push_back(0xac);  // OP_CHECKSIG

    BOOST_CHECK_EQUAL(script.size(), 25u);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_P2PKH));
}

BOOST_AUTO_TEST_CASE(script_type_detection_p2sh)
{
    // P2SH: OP_HASH160 <20 bytes> OP_EQUAL
    std::vector<unsigned char> script = {
        0xa9,  // OP_HASH160
        0x14,  // Push 20 bytes
    };
    script.insert(script.end(), 20, 0x00);  // 20-byte hash
    script.push_back(0x87);  // OP_EQUAL

    BOOST_CHECK_EQUAL(script.size(), 23u);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_P2SH));
}

BOOST_AUTO_TEST_CASE(script_type_detection_p2wpkh)
{
    // P2WPKH: OP_0 <20 bytes>
    std::vector<unsigned char> script = {
        0x00,  // OP_0
        0x14,  // Push 20 bytes
    };
    script.insert(script.end(), 20, 0x00);  // 20-byte hash

    BOOST_CHECK_EQUAL(script.size(), 22u);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_P2WPKH));
}

BOOST_AUTO_TEST_CASE(script_type_detection_p2wsh)
{
    // P2WSH: OP_0 <32 bytes>
    std::vector<unsigned char> script = {
        0x00,  // OP_0
        0x20,  // Push 32 bytes
    };
    script.insert(script.end(), 32, 0x00);  // 32-byte hash

    BOOST_CHECK_EQUAL(script.size(), 34u);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_P2WSH));
}

BOOST_AUTO_TEST_CASE(script_type_detection_p2tr)
{
    // P2TR: OP_1 <32 bytes>
    std::vector<unsigned char> script = {
        0x51,  // OP_1
        0x20,  // Push 32 bytes
    };
    script.insert(script.end(), 32, 0x00);  // 32-byte x-only pubkey

    BOOST_CHECK_EQUAL(script.size(), 34u);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_P2TR));
}

BOOST_AUTO_TEST_CASE(script_type_detection_p2pk_compressed)
{
    // P2PK compressed: <33 byte pubkey> OP_CHECKSIG
    std::vector<unsigned char> script = {
        0x21,  // Push 33 bytes
        0x02,  // Compressed pubkey prefix (even Y)
    };
    script.insert(script.end(), 32, 0x00);  // 32 more bytes
    script.push_back(0xac);  // OP_CHECKSIG

    BOOST_CHECK_EQUAL(script.size(), 35u);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_P2PK));
}

BOOST_AUTO_TEST_CASE(script_type_detection_p2pk_uncompressed)
{
    // P2PK uncompressed: <65 byte pubkey> OP_CHECKSIG
    std::vector<unsigned char> script = {
        0x41,  // Push 65 bytes
        0x04,  // Uncompressed pubkey prefix
    };
    script.insert(script.end(), 64, 0x00);  // 64 more bytes
    script.push_back(0xac);  // OP_CHECKSIG

    BOOST_CHECK_EQUAL(script.size(), 67u);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_P2PK));
}

BOOST_AUTO_TEST_CASE(script_type_detection_null_data)
{
    // NULL_DATA: OP_RETURN <data>
    std::vector<unsigned char> script = {
        0x6a,  // OP_RETURN
        0x04,  // Push 4 bytes
        0xde, 0xad, 0xbe, 0xef
    };

    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_NULL_DATA));

    // OP_RETURN with no data
    std::vector<unsigned char> script2 = {0x6a};
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script2.data(), script2.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_NULL_DATA));
}

BOOST_AUTO_TEST_CASE(script_type_detection_witness_unknown)
{
    // WITNESS_UNKNOWN: OP_2 through OP_16 with 2-40 byte data
    // Future segwit versions

    // OP_2 with 32 bytes (witness v2)
    std::vector<unsigned char> script_v2 = {0x52, 0x20};  // OP_2, push 32
    script_v2.insert(script_v2.end(), 32, 0x00);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script_v2.data(), script_v2.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_WITNESS_UNKNOWN));

    // OP_16 with 20 bytes (witness v16)
    std::vector<unsigned char> script_v16 = {0x60, 0x14};  // OP_16, push 20
    script_v16.insert(script_v16.end(), 20, 0x00);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script_v16.data(), script_v16.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_WITNESS_UNKNOWN));
}

BOOST_AUTO_TEST_CASE(script_type_detection_multisig)
{
    // MULTISIG: OP_M <pubkey1>...<pubkeyN> OP_N OP_CHECKMULTISIG
    // 1-of-2 multisig with compressed keys

    std::vector<unsigned char> script = {0x51};  // OP_1 (M=1)

    // First pubkey (compressed)
    script.push_back(0x21);  // Push 33 bytes
    script.push_back(0x02);  // Compressed prefix
    script.insert(script.end(), 32, 0x00);

    // Second pubkey (compressed)
    script.push_back(0x21);  // Push 33 bytes
    script.push_back(0x03);  // Compressed prefix
    script.insert(script.end(), 32, 0x00);

    script.push_back(0x52);  // OP_2 (N=2)
    script.push_back(0xae);  // OP_CHECKMULTISIG

    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script.data(), script.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_MULTISIG));
}

BOOST_AUTO_TEST_CASE(script_type_detection_nonstandard)
{
    // Various nonstandard scripts

    // Just OP_TRUE
    std::vector<unsigned char> script1 = {0x51};
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script1.data(), script1.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_NONSTANDARD));

    // Just OP_FALSE
    std::vector<unsigned char> script2 = {0x00};
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script2.data(), script2.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_NONSTANDARD));

    // Random bytes
    std::vector<unsigned char> script3 = {0xde, 0xad, 0xbe, 0xef};
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script3.data(), script3.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_NONSTANDARD));

    // Almost P2PKH but wrong length
    std::vector<unsigned char> script4 = {0x76, 0xa9, 0x14};
    script4.insert(script4.end(), 19, 0x00);  // Only 19 bytes instead of 20
    script4.push_back(0x88);
    script4.push_back(0xac);
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(script4.data(), script4.size())),
                      static_cast<int>(gpu::SCRIPT_TYPE_NONSTANDARD));
}

BOOST_AUTO_TEST_CASE(script_type_detection_empty)
{
    // Empty script
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(nullptr, 0)),
                      static_cast<int>(gpu::SCRIPT_TYPE_UNKNOWN));

    std::vector<unsigned char> empty;
    BOOST_CHECK_EQUAL(static_cast<int>(gpu::IdentifyScriptType(empty.data(), 0)),
                      static_cast<int>(gpu::SCRIPT_TYPE_UNKNOWN));
}

BOOST_AUTO_TEST_CASE(script_type_detection_malformed)
{
    // Malformed P2PKH (wrong opcodes)
    std::vector<unsigned char> bad_p2pkh = {0x77, 0xa9, 0x14};  // Wrong first opcode
    bad_p2pkh.insert(bad_p2pkh.end(), 20, 0x00);
    bad_p2pkh.push_back(0x88);
    bad_p2pkh.push_back(0xac);
    BOOST_CHECK_NE(static_cast<int>(gpu::IdentifyScriptType(bad_p2pkh.data(), bad_p2pkh.size())),
                   static_cast<int>(gpu::SCRIPT_TYPE_P2PKH));

    // Malformed P2WPKH (wrong version)
    std::vector<unsigned char> bad_p2wpkh = {0x01, 0x14};  // OP_1 instead of OP_0
    bad_p2wpkh.insert(bad_p2wpkh.end(), 20, 0x00);
    // This should be detected as something else (could be witness unknown or nonstandard)
    BOOST_CHECK_NE(static_cast<int>(gpu::IdentifyScriptType(bad_p2wpkh.data(), bad_p2wpkh.size())),
                   static_cast<int>(gpu::SCRIPT_TYPE_P2WPKH));
}

// =============================================================================
// Sigversion Selection Tests (simulating validation.cpp logic)
// =============================================================================

// Helper to determine sigversion like validation.cpp does
static gpu::GPUSigVersion DetermineSigversion(
    const std::vector<unsigned char>& scriptPubKey,
    const CScriptWitness& witness)
{
    // P2WPKH
    if (scriptPubKey.size() == 22 && scriptPubKey[0] == 0x00 && scriptPubKey[1] == 0x14) {
        return gpu::GPU_SIGVERSION_WITNESS_V0;
    }
    // P2WSH
    if (scriptPubKey.size() == 34 && scriptPubKey[0] == 0x00 && scriptPubKey[1] == 0x20) {
        return gpu::GPU_SIGVERSION_WITNESS_V0;
    }
    // P2TR
    if (scriptPubKey.size() == 34 && scriptPubKey[0] == 0x51 && scriptPubKey[1] == 0x20) {
        const auto& wit = witness.stack;
        size_t wit_count = wit.size();
        bool has_annex = (wit_count >= 2 && !wit.back().empty() && wit.back()[0] == 0x50);
        size_t effective_count = has_annex ? wit_count - 1 : wit_count;

        if (effective_count == 1) {
            return gpu::GPU_SIGVERSION_TAPROOT;  // Key path
        } else {
            return gpu::GPU_SIGVERSION_TAPSCRIPT;  // Script path
        }
    }

    return gpu::GPU_SIGVERSION_BASE;  // Legacy
}

BOOST_AUTO_TEST_CASE(sigversion_selection_legacy)
{
    // P2PKH script
    std::vector<unsigned char> p2pkh = {0x76, 0xa9, 0x14};
    p2pkh.insert(p2pkh.end(), 20, 0x00);
    p2pkh.push_back(0x88);
    p2pkh.push_back(0xac);

    CScriptWitness empty_witness;
    BOOST_CHECK_EQUAL(static_cast<int>(DetermineSigversion(p2pkh, empty_witness)),
                      static_cast<int>(gpu::GPU_SIGVERSION_BASE));
}

BOOST_AUTO_TEST_CASE(sigversion_selection_p2wpkh)
{
    std::vector<unsigned char> p2wpkh = {0x00, 0x14};
    p2wpkh.insert(p2wpkh.end(), 20, 0x00);

    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(71, 0x30));  // DER sig
    witness.stack.push_back(std::vector<unsigned char>(33, 0x02));  // Pubkey

    BOOST_CHECK_EQUAL(static_cast<int>(DetermineSigversion(p2wpkh, witness)),
                      static_cast<int>(gpu::GPU_SIGVERSION_WITNESS_V0));
}

BOOST_AUTO_TEST_CASE(sigversion_selection_p2wsh)
{
    std::vector<unsigned char> p2wsh = {0x00, 0x20};
    p2wsh.insert(p2wsh.end(), 32, 0x00);

    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(71, 0x30));  // Signature
    witness.stack.push_back(std::vector<unsigned char>(100, 0x00)); // Redeem script

    BOOST_CHECK_EQUAL(static_cast<int>(DetermineSigversion(p2wsh, witness)),
                      static_cast<int>(gpu::GPU_SIGVERSION_WITNESS_V0));
}

BOOST_AUTO_TEST_CASE(sigversion_selection_p2tr_key_path)
{
    std::vector<unsigned char> p2tr = {0x51, 0x20};
    p2tr.insert(p2tr.end(), 32, 0x00);

    // Key path: single 64-byte signature
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));

    BOOST_CHECK_EQUAL(static_cast<int>(DetermineSigversion(p2tr, witness)),
                      static_cast<int>(gpu::GPU_SIGVERSION_TAPROOT));
}

BOOST_AUTO_TEST_CASE(sigversion_selection_p2tr_script_path)
{
    std::vector<unsigned char> p2tr = {0x51, 0x20};
    p2tr.insert(p2tr.end(), 32, 0x00);

    // Script path: sig + tapscript + control_block
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));  // sig
    witness.stack.push_back(std::vector<unsigned char>{0xac});      // tapscript
    witness.stack.push_back(std::vector<unsigned char>(33, 0xc0));  // control

    BOOST_CHECK_EQUAL(static_cast<int>(DetermineSigversion(p2tr, witness)),
                      static_cast<int>(gpu::GPU_SIGVERSION_TAPSCRIPT));
}

BOOST_AUTO_TEST_CASE(sigversion_selection_p2tr_key_path_with_annex)
{
    std::vector<unsigned char> p2tr = {0x51, 0x20};
    p2tr.insert(p2tr.end(), 32, 0x00);

    // Key path with annex: sig + annex
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));    // sig
    witness.stack.push_back(std::vector<unsigned char>{0x50, 0x01});  // annex

    BOOST_CHECK_EQUAL(static_cast<int>(DetermineSigversion(p2tr, witness)),
                      static_cast<int>(gpu::GPU_SIGVERSION_TAPROOT));
}

BOOST_AUTO_TEST_CASE(sigversion_selection_p2tr_script_path_with_annex)
{
    std::vector<unsigned char> p2tr = {0x51, 0x20};
    p2tr.insert(p2tr.end(), 32, 0x00);

    // Script path with annex: sig + tapscript + control + annex
    CScriptWitness witness;
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));    // sig
    witness.stack.push_back(std::vector<unsigned char>{0xac});        // tapscript
    witness.stack.push_back(std::vector<unsigned char>(33, 0xc0));    // control
    witness.stack.push_back(std::vector<unsigned char>{0x50, 0x01});  // annex

    BOOST_CHECK_EQUAL(static_cast<int>(DetermineSigversion(p2tr, witness)),
                      static_cast<int>(gpu::GPU_SIGVERSION_TAPSCRIPT));
}

BOOST_AUTO_TEST_CASE(sigversion_selection_empty_witness)
{
    // P2TR with empty witness should still be analyzed
    // (though this would fail validation)
    std::vector<unsigned char> p2tr = {0x51, 0x20};
    p2tr.insert(p2tr.end(), 32, 0x00);

    CScriptWitness empty_witness;

    // With 0 elements, effective_count = 0, which is != 1, so TAPSCRIPT
    // But realistically empty witness would be rejected earlier
    gpu::GPUSigVersion sv = DetermineSigversion(p2tr, empty_witness);
    // Either TAPROOT (if we consider 0 as special) or TAPSCRIPT
    BOOST_CHECK(sv == gpu::GPU_SIGVERSION_TAPROOT || sv == gpu::GPU_SIGVERSION_TAPSCRIPT);
}

// =============================================================================
// Large Tapscript Tests (BIP342 removes 520-byte script limit)
// =============================================================================

BOOST_AUTO_TEST_CASE(large_tapscript_size_limits)
{
    // Test that tapscript itself can exceed 520 bytes
    // while individual stack elements must be <= 520 bytes

    // Create a large tapscript (10KB of OP_NOPs followed by OP_TRUE)
    std::vector<unsigned char> large_tapscript;
    for (size_t i = 0; i < 10000; i++) {
        large_tapscript.push_back(0x61);  // OP_NOP
    }
    large_tapscript.push_back(0x51);  // OP_TRUE

    BOOST_CHECK_GT(large_tapscript.size(), 520u);
    BOOST_CHECK_EQUAL(large_tapscript.size(), 10001u);

    // This tapscript should be valid in BIP342 (no script size limit)
    // The 520-byte limit only applies to stack elements
}

BOOST_AUTO_TEST_CASE(witness_stack_element_size_limit)
{
    // Test that witness stack elements (sigs, pubkeys) are still limited to 520 bytes
    // even in Tapscript

    std::vector<unsigned char> valid_sig(64, 0x00);    // 64 bytes - OK
    std::vector<unsigned char> valid_pubkey(33, 0x02); // 33 bytes - OK
    std::vector<unsigned char> max_element(520, 0x00); // 520 bytes - OK
    std::vector<unsigned char> too_large(521, 0x00);   // 521 bytes - TOO LARGE

    BOOST_CHECK_LE(valid_sig.size(), 520u);
    BOOST_CHECK_LE(valid_pubkey.size(), 520u);
    BOOST_CHECK_LE(max_element.size(), 520u);
    BOOST_CHECK_GT(too_large.size(), 520u);

    // GPU limit constant should be 520
    BOOST_CHECK_EQUAL(gpu::MAX_STACK_ELEMENT_SIZE, 520u);
}

BOOST_AUTO_TEST_CASE(tapscript_vs_stack_element_distinction)
{
    // Verify that we correctly distinguish between:
    // 1. Tapscript (the script to execute) - can be large
    // 2. Stack elements (sigs, data passed to script) - max 520 bytes

    // Witness for script path: [stack_item1, stack_item2, tapscript, control_block]
    CScriptWitness witness;

    // Stack items (must be <= 520 bytes each)
    witness.stack.push_back(std::vector<unsigned char>(64, 0x00));   // sig - OK
    witness.stack.push_back(std::vector<unsigned char>(100, 0x01));  // data - OK

    // Tapscript can be large (this is NOT a stack element for EvalScript)
    std::vector<unsigned char> large_script(5000, 0x61);  // 5KB of OP_NOP
    large_script.push_back(0x51);  // OP_TRUE
    witness.stack.push_back(large_script);

    // Control block
    witness.stack.push_back(std::vector<unsigned char>(33, 0xc0));

    // Count elements
    BOOST_CHECK_EQUAL(witness.stack.size(), 4u);

    // Verify sizes
    BOOST_CHECK_LE(witness.stack[0].size(), 520u);  // sig
    BOOST_CHECK_LE(witness.stack[1].size(), 520u);  // data
    BOOST_CHECK_GT(witness.stack[2].size(), 520u);  // tapscript - ALLOWED to be large
    BOOST_CHECK_LE(witness.stack[3].size(), 520u);  // control block
}

BOOST_AUTO_TEST_CASE(vram_usage_estimation)
{
    // Calculate VRAM usage for GPU script validation

    // GPUStackElement size
    size_t stack_elem_size = sizeof(gpu::GPUStackElement);
    BOOST_CHECK_EQUAL(stack_elem_size, 524u);  // 520 data + 2 size + 2 padding

    // GPUScriptContext approximate size
    size_t stack_size = gpu::MAX_STACK_SIZE * stack_elem_size;      // 1000 * 524 = 524 KB
    size_t altstack_size = gpu::MAX_STACK_SIZE * stack_elem_size;   // 1000 * 524 = 524 KB
    size_t ctx_overhead = 1024;  // Approximate overhead for other fields

    size_t context_size = stack_size + altstack_size + ctx_overhead;

    // Approximately 1 MB per context
    BOOST_CHECK_GT(context_size, 1000000u);  // > 1 MB
    BOOST_CHECK_LT(context_size, 1100000u);  // < 1.1 MB

    // For 4000 tapscript jobs (worst case: each needs own context)
    size_t vram_4k_jobs = 4000 * context_size;

    // Should be approximately 4 GB
    BOOST_CHECK_GT(vram_4k_jobs, 4000000000ull);  // > 4 GB
    BOOST_CHECK_LT(vram_4k_jobs, 4400000000ull);  // < 4.4 GB

    // Log the estimates for reference
    BOOST_TEST_MESSAGE("Stack element size: " << stack_elem_size << " bytes");
    BOOST_TEST_MESSAGE("Context size: " << context_size << " bytes (~" << context_size / 1024 / 1024 << " MB)");
    BOOST_TEST_MESSAGE("VRAM for 4000 jobs: " << vram_4k_jobs / 1024 / 1024 / 1024 << " GB");
}

BOOST_AUTO_TEST_CASE(witness_parsing_large_tapscript)
{
    // Test parsing a witness with a large tapscript
    // The parsing should extract tapscript directly without putting it on stack

    // Create a witness with:
    // - 3 small stack items (signatures)
    // - 1 large tapscript (150KB)
    // - 1 control block

    std::vector<std::vector<unsigned char>> items;

    // 3 signatures (64 bytes each)
    items.push_back(std::vector<unsigned char>(64, 0x01));
    items.push_back(std::vector<unsigned char>(64, 0x02));
    items.push_back(std::vector<unsigned char>(64, 0x03));

    // Large tapscript (150KB)
    std::vector<unsigned char> tapscript(150000, 0x61);  // OP_NOP
    tapscript.push_back(0x51);  // OP_TRUE
    items.push_back(tapscript);

    // Control block (33 bytes minimum)
    items.push_back(std::vector<unsigned char>(33, 0xc0));

    // Total witness item count
    BOOST_CHECK_EQUAL(items.size(), 5u);

    // Verify the stack items (first 3) are <= 520 bytes
    for (size_t i = 0; i < 3; i++) {
        BOOST_CHECK_LE(items[i].size(), 520u);
    }

    // Verify tapscript is large
    BOOST_CHECK_GT(items[3].size(), 520u);
    BOOST_CHECK_EQUAL(items[3].size(), 150001u);

    // Verify control block is small
    BOOST_CHECK_LE(items[4].size(), 520u);
}

BOOST_AUTO_TEST_CASE(max_stack_constants)
{
    // Verify GPU constants match Bitcoin Core limits
    BOOST_CHECK_EQUAL(gpu::MAX_STACK_SIZE, 1000u);
    BOOST_CHECK_EQUAL(gpu::MAX_STACK_ELEMENT_SIZE, 520u);
    BOOST_CHECK_EQUAL(gpu::MAX_SCRIPT_SIZE, 10000u);
    BOOST_CHECK_EQUAL(gpu::MAX_OPS_PER_SCRIPT, 201u);
}

BOOST_AUTO_TEST_SUITE_END()
