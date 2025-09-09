// Copyright (c) 2024-present The Bitcoin Core developers  
// GPU Hash Function Tests - Complete coverage for SipHash implementation

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <gpu_kernel/gpu_hash.h>
#include <gpu_kernel/gpu_types.h>
#include <gpu_kernel/gpu_utils.h>
#include <gpu_kernel/gpu_utxo.h>
#include <crypto/siphash.h>
#include <random.h>
#include <uint256.h>

BOOST_AUTO_TEST_SUITE(gpu_hash_tests)

BOOST_FIXTURE_TEST_CASE(gpu_siphash_correctness, BasicTestingSetup)
{
    // Test that GPU SipHash matches CPU implementation
    for (int i = 0; i < 100; i++) {
        uint256 test_val = GetRandHash();
        uint64_t k0 = m_rng.rand64();
        uint64_t k1 = m_rng.rand64();
        
        // CPU hash
        uint64_t cpu_hash = SipHashUint256(k0, k1, test_val);
        
        // GPU hash (host-callable version)
        gpu::uint256_gpu gpu_val = gpu::ToGPU(test_val);
        uint64_t gpu_hash = gpu::SipHashUint256(k0, k1, gpu_val);
        
        BOOST_CHECK_EQUAL(cpu_hash, gpu_hash);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_siphash_extra_correctness, BasicTestingSetup)
{
    // Test SipHash with extra data (for outpoints)
    for (int i = 0; i < 100; i++) {
        uint256 test_val = GetRandHash();
        uint32_t extra = m_rng.rand32();
        uint64_t k0 = m_rng.rand64();
        uint64_t k1 = m_rng.rand64();
        
        // CPU hash
        uint64_t cpu_hash = SipHashUint256Extra(k0, k1, test_val, extra);
        
        // GPU hash
        gpu::uint256_gpu gpu_val = gpu::ToGPU(test_val);
        uint64_t gpu_hash = gpu::SipHashUint256Extra(k0, k1, gpu_val, extra);
        
        BOOST_CHECK_EQUAL(cpu_hash, gpu_hash);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_distribution, BasicTestingSetup)
{
    // Test hash distribution quality
    const int NUM_SAMPLES = 10000;
    const int NUM_BUCKETS = 256;
    std::vector<int> buckets(NUM_BUCKETS, 0);
    
    uint64_t k0 = m_rng.rand64();
    uint64_t k1 = m_rng.rand64();
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        uint256 val = GetRandHash();
        gpu::uint256_gpu gpu_val = gpu::ToGPU(val);
        uint64_t hash = gpu::SipHashUint256(k0, k1, gpu_val);
        
        buckets[hash % NUM_BUCKETS]++;
    }
    
    // Check for reasonable distribution (chi-square test)
    double expected = static_cast<double>(NUM_SAMPLES) / NUM_BUCKETS;
    double chi_square = 0;
    
    for (int count : buckets) {
        double diff = count - expected;
        chi_square += (diff * diff) / expected;
    }
    
    // With 255 degrees of freedom, chi-square should be < 310 for p=0.01
    BOOST_CHECK_LT(chi_square, 310);
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_avalanche_effect, BasicTestingSetup)
{
    // Test that small input changes cause large hash changes
    uint256 base = GetRandHash();
    uint64_t k0 = m_rng.rand64();
    uint64_t k1 = m_rng.rand64();
    
    gpu::uint256_gpu gpu_base = gpu::ToGPU(base);
    uint64_t base_hash = gpu::SipHashUint256(k0, k1, gpu_base);
    
    // Flip each bit and check hash difference
    for (int byte = 0; byte < 32; byte++) {
        for (int bit = 0; bit < 8; bit++) {
            uint256 modified = base;
            modified.begin()[byte] ^= (1 << bit);
            
            gpu::uint256_gpu gpu_modified = gpu::ToGPU(modified);
            uint64_t modified_hash = gpu::SipHashUint256(k0, k1, gpu_modified);
            
            // Count different bits
            uint64_t diff = base_hash ^ modified_hash;
            int bit_diff = __builtin_popcountll(diff);
            
            // Should change roughly half the bits (32 ± tolerance)
            BOOST_CHECK_GT(bit_diff, 20);
            BOOST_CHECK_LT(bit_diff, 44);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_4way_hash_independence, BasicTestingSetup)
{
    // Test that the 4 hash functions are independent
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    const int NUM_TESTS = 1000;
    std::set<uint32_t> hash1_values, hash2_values, hash3_values, hash4_values;
    
    for (int i = 0; i < NUM_TESTS; i++) {
        uint256 txid = GetRandHash();
        uint32_t vout = m_rng.rand32();
        
        uint32_t h1 = utxoSet.Hash1(txid, vout);
        uint32_t h2 = utxoSet.Hash2(txid, vout);
        uint32_t h3 = utxoSet.Hash3(txid, vout);
        uint32_t h4 = utxoSet.Hash4(txid, vout);
        
        // All should be different
        BOOST_CHECK_NE(h1, h2);
        BOOST_CHECK_NE(h1, h3);
        BOOST_CHECK_NE(h1, h4);
        BOOST_CHECK_NE(h2, h3);
        BOOST_CHECK_NE(h2, h4);
        BOOST_CHECK_NE(h3, h4);
        
        hash1_values.insert(h1);
        hash2_values.insert(h2);
        hash3_values.insert(h3);
        hash4_values.insert(h4);
    }
    
    // Check uniqueness (should have many unique values)
    BOOST_CHECK_GT(hash1_values.size(), NUM_TESTS * 0.95);
    BOOST_CHECK_GT(hash2_values.size(), NUM_TESTS * 0.95);
    BOOST_CHECK_GT(hash3_values.size(), NUM_TESTS * 0.95);
    BOOST_CHECK_GT(hash4_values.size(), NUM_TESTS * 0.95);
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_determinism, BasicTestingSetup)
{
    // Test that hashes are deterministic
    uint256 txid = GetRandHash();
    uint32_t vout = 42;
    uint64_t k0 = 0x0123456789ABCDEF;
    uint64_t k1 = 0xFEDCBA9876543210;
    
    gpu::uint256_gpu gpu_txid = gpu::ToGPU(txid);
    
    // Hash multiple times - should always get same result
    uint64_t hash1 = gpu::SipHashUint256Extra(k0, k1, gpu_txid, vout);
    uint64_t hash2 = gpu::SipHashUint256Extra(k0, k1, gpu_txid, vout);
    uint64_t hash3 = gpu::SipHashUint256Extra(k0, k1, gpu_txid, vout);
    
    BOOST_CHECK_EQUAL(hash1, hash2);
    BOOST_CHECK_EQUAL(hash2, hash3);
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_edge_cases, BasicTestingSetup)
{
    // Test edge cases
    uint64_t k0 = m_rng.rand64();
    uint64_t k1 = m_rng.rand64();
    
    // All zeros
    uint256 zero;
    gpu::uint256_gpu gpu_zero = gpu::ToGPU(zero);
    uint64_t zero_hash = gpu::SipHashUint256(k0, k1, gpu_zero);
    BOOST_CHECK_NE(zero_hash, 0); // Should not produce zero
    
    // All ones
    uint256 ones;
    memset(ones.begin(), 0xFF, 32);
    gpu::uint256_gpu gpu_ones = gpu::ToGPU(ones);
    uint64_t ones_hash = gpu::SipHashUint256(k0, k1, gpu_ones);
    BOOST_CHECK_NE(ones_hash, 0xFFFFFFFFFFFFFFFF); // Should not produce all ones
    
    // Should be different
    BOOST_CHECK_NE(zero_hash, ones_hash);
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_collision_resistance, BasicTestingSetup)
{
    // Test collision resistance within table size
    const int NUM_SAMPLES = 100000;
    std::set<uint32_t> seen_hashes;
    
    gpu::GPUUTXOSet utxoSet;
    BOOST_REQUIRE(utxoSet.Initialize());
    
    int collisions = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        uint256 txid = GetRandHash();
        uint32_t vout = m_rng.rand32();
        
        uint32_t hash = utxoSet.Hash1(txid, vout);
        
        if (seen_hashes.find(hash) != seen_hashes.end()) {
            collisions++;
        }
        seen_hashes.insert(hash);
    }
    
    // Expect some collisions due to birthday paradox, but not too many
    double collision_rate = static_cast<double>(collisions) / NUM_SAMPLES;
    BOOST_CHECK_LT(collision_rate, 0.01); // Less than 1% collision rate
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_performance_characteristics, BasicTestingSetup)
{
    // Test that hash computation is fast
    const int NUM_ITERATIONS = 1000000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t k0 = m_rng.rand64();
    uint64_t k1 = m_rng.rand64();
    uint256 val = GetRandHash();
    gpu::uint256_gpu gpu_val = gpu::ToGPU(val);
    
    uint64_t sum = 0; // Prevent optimization
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        sum += gpu::SipHashUint256(k0, k1, gpu_val);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double hashes_per_second = (NUM_ITERATIONS * 1000000.0) / duration.count();
    
    BOOST_TEST_MESSAGE("Hash performance: " + std::to_string(hashes_per_second / 1000000) + " MH/s");
    
    // Should achieve at least 10 MH/s on CPU (GPU would be much faster)
    BOOST_CHECK_GT(hashes_per_second, 10000000);
    
    // Use sum to prevent optimization
    BOOST_CHECK_NE(sum, 0);
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_salt_sensitivity, BasicTestingSetup)
{
    // Test that different salts produce different hashes
    uint256 txid = GetRandHash();
    uint32_t vout = 42;
    gpu::uint256_gpu gpu_txid = gpu::ToGPU(txid);
    
    std::set<uint64_t> hashes;
    
    // Try many different salt pairs
    for (int i = 0; i < 1000; i++) {
        uint64_t k0 = m_rng.rand64();
        uint64_t k1 = m_rng.rand64();
        
        uint64_t hash = gpu::SipHashUint256Extra(k0, k1, gpu_txid, vout);
        hashes.insert(hash);
    }
    
    // All should be unique (extremely unlikely to have collisions)
    BOOST_CHECK_EQUAL(hashes.size(), 1000);
}

BOOST_AUTO_TEST_SUITE_END()