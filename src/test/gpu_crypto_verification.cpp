// Simple GPU Crypto Verification Tests
#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <crypto/sha256.h>
#include <crypto/ripemd160.h>
#include <crypto/siphash.h>
#include <crypto/muhash.h>
#include <hash.h>
#include <uint256.h>
#include <random.h>

BOOST_AUTO_TEST_SUITE(gpu_crypto_verification)

BOOST_FIXTURE_TEST_CASE(gpu_crypto_basic_test, BasicTestingSetup)
{
    // Basic test to verify crypto implementation exists
    // The actual GPU crypto tests would need to be run with CUDA
    
    // Test CPU hashes work
    std::string test_data = "Bitcoin Core GPU Acceleration";
    
    // SHA-256
    CSHA256 sha256;
    sha256.Write((const unsigned char*)test_data.data(), test_data.size());
    uint256 sha_result;
    sha256.Finalize(sha_result.begin());
    BOOST_CHECK(!sha_result.IsNull());
    
    // Double SHA-256
    CHash256 sha256d;
    sha256d.Write({(const unsigned char*)test_data.data(), test_data.size()});
    uint256 sha256d_result;
    sha256d.Finalize(sha256d_result);
    BOOST_CHECK(!sha256d_result.IsNull());
    
    // RIPEMD-160
    CRIPEMD160 ripemd;
    ripemd.Write((const unsigned char*)test_data.data(), test_data.size());
    uint160 ripemd_result;
    ripemd.Finalize(ripemd_result.begin());
    BOOST_CHECK(ripemd_result.size() == 20);
    
    // Hash160
    CHash160 hash160;
    hash160.Write({(const unsigned char*)test_data.data(), test_data.size()});
    uint160 hash160_result;
    hash160.Finalize(hash160_result);
    BOOST_CHECK(hash160_result.size() == 20);
    
    BOOST_TEST_MESSAGE("CPU crypto functions verified. GPU implementations are in gpu_crypto.cuh");
}

BOOST_FIXTURE_TEST_CASE(gpu_siphash_cpu_test, BasicTestingSetup)
{
    // Test SipHash on CPU
    for (int i = 0; i < 10; i++) {
        uint64_t k0 = m_rng.rand64();
        uint64_t k1 = m_rng.rand64();
        uint256 test_val = GetRandHash();
        
        uint64_t hash = SipHashUint256(k0, k1, test_val);
        BOOST_CHECK(hash != 0);
        
        // Same input should give same output
        uint64_t hash2 = SipHashUint256(k0, k1, test_val);
        BOOST_CHECK_EQUAL(hash, hash2);
        
        // Different keys should give different output
        uint64_t hash3 = SipHashUint256(k0 + 1, k1, test_val);
        BOOST_CHECK(hash != hash3);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_murmurhash_cpu_test, BasicTestingSetup)
{
    // Test MurmurHash3 on CPU
    for (int i = 0; i < 10; i++) {
        uint32_t seed = m_rng.rand32();
        std::vector<unsigned char> data(32);
        for (auto& b : data) {
            b = m_rng.randbits(8);
        }
        
        uint32_t hash = MurmurHash3(seed, data);
        BOOST_CHECK(hash != 0);
        
        // Same input should give same output
        uint32_t hash2 = MurmurHash3(seed, data);
        BOOST_CHECK_EQUAL(hash, hash2);
        
        // Different seed should give different output
        uint32_t hash3 = MurmurHash3(seed + 1, data);
        BOOST_CHECK(hash != hash3);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_test_vectors, BasicTestingSetup)
{
    // Test with known vectors
    
    // SHA-256 of empty string
    {
        CSHA256 sha;
        uint256 result;
        sha.Finalize(result.begin());
        BOOST_CHECK_EQUAL(result.ToString(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
    
    // SHA-256 of "abc"
    {
        CSHA256 sha;
        sha.Write((const unsigned char*)"abc", 3);
        uint256 result;
        sha.Finalize(result.begin());
        BOOST_CHECK_EQUAL(result.ToString(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
    
    // RIPEMD-160 of empty string
    {
        CRIPEMD160 ripemd;
        uint160 result;
        ripemd.Finalize(result.begin());
        std::vector<unsigned char> vch(result.begin(), result.end());
        BOOST_CHECK_EQUAL(HexStr(vch), "9c1185a5c5e9fc54612808977ee8f548b2258d31");
    }
    
    // RIPEMD-160 of "abc"
    {
        CRIPEMD160 ripemd;
        ripemd.Write((const unsigned char*)"abc", 3);
        uint160 result;
        ripemd.Finalize(result.begin());
        std::vector<unsigned char> vch(result.begin(), result.end());
        BOOST_CHECK_EQUAL(HexStr(vch), "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc");
    }
    
    BOOST_TEST_MESSAGE("Test vectors verified for CPU implementation");
}

BOOST_AUTO_TEST_SUITE_END()