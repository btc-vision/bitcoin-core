// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <crypto/sha256.h>
#include <crypto/sha1.h>
#include <crypto/ripemd160.h>
#include <crypto/siphash.h>
#include <hash.h>
#include <uint256.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

// External functions from gpu_hash_test_kernels.cu
extern "C" {
    void LaunchSHA256Test(const uint8_t* input, size_t len, uint8_t* output);
    void LaunchSHA256dTest(const uint8_t* input, size_t len, uint8_t* output);
    void LaunchSHA1Test(const uint8_t* input, size_t len, uint8_t* output);
    void LaunchRIPEMD160Test(const uint8_t* input, size_t len, uint8_t* output);
    void LaunchHash160Test(const uint8_t* input, size_t len, uint8_t* output);
    void LaunchSipHashTest(uint64_t k0, uint64_t k1, const uint8_t* data, size_t len, uint64_t* output);
    void LaunchMurmurHash3Test(uint32_t seed, const uint8_t* data, size_t len, uint32_t* output);
}

BOOST_FIXTURE_TEST_SUITE(gpu_hash_verification_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(gpu_sha256_verification)
{
    // Test vectors
    std::vector<std::pair<std::string, std::string>> test_vectors = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"message digest", "f7846f55cf23e14eebeab5b4e1550cad5b509e3348fbc4efa3a1413d393cb650"},
        {"The quick brown fox jumps over the lazy dog", "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"}
    };
    
    for (const auto& [input_str, expected_hex] : test_vectors) {
        std::vector<uint8_t> input(input_str.begin(), input_str.end());
        
        // CPU computation
        uint8_t cpu_hash[32];
        CSHA256().Write(input.data(), input.size()).Finalize(cpu_hash);
        
        // GPU computation
        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        
        cudaMalloc(&d_input, std::max(input.size(), size_t(1)));
        cudaMalloc(&d_output, 32);
        
        if (!input.empty()) {
            cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice);
        }
        
        LaunchSHA256Test(d_input, input.size(), d_output);
        
        uint8_t gpu_hash[32];
        cudaMemcpy(gpu_hash, d_output, 32, cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        // Compare results
        BOOST_CHECK_MESSAGE(
            memcmp(cpu_hash, gpu_hash, 32) == 0,
            "SHA256 mismatch for input: \"" + input_str + "\""
        );
    }
}

BOOST_AUTO_TEST_CASE(gpu_sha256d_verification)
{
    for (int test = 0; test < 100; test++) {
        size_t len = m_rng.randrange(200);
        std::vector<uint8_t> data(len);
        for (size_t i = 0; i < len; i++) {
            data[i] = m_rng.randbits(8);
        }
        
        // CPU computation
        uint8_t cpu_hash[32];
        CHash256().Write({data.data(), len}).Finalize({cpu_hash});
        
        // GPU computation
        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        
        cudaMalloc(&d_input, std::max(len, size_t(1)));
        cudaMalloc(&d_output, 32);
        
        if (!data.empty()) {
            cudaMemcpy(d_input, data.data(), len, cudaMemcpyHostToDevice);
        }
        
        LaunchSHA256dTest(d_input, len, d_output);
        
        uint8_t gpu_hash[32];
        cudaMemcpy(gpu_hash, d_output, 32, cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        BOOST_CHECK_MESSAGE(
            memcmp(cpu_hash, gpu_hash, 32) == 0,
            "SHA256d mismatch for test " + std::to_string(test)
        );
    }
}

BOOST_AUTO_TEST_CASE(gpu_sha1_verification)
{
    // Test vectors
    std::vector<std::pair<std::string, std::string>> test_vectors = {
        {"", "da39a3ee5e6b4b0d3255bfef95601890afd80709"},
        {"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"},
        {"message digest", "c12252ceda8be8994d5fa0290a47231c1d16aae3"},
        {"The quick brown fox jumps over the lazy dog", "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"}
    };
    
    for (const auto& [input_str, expected_hex] : test_vectors) {
        std::vector<uint8_t> input(input_str.begin(), input_str.end());
        
        // CPU computation
        uint8_t cpu_hash[20];
        CSHA1().Write(input.data(), input.size()).Finalize(cpu_hash);
        
        // GPU computation
        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        
        cudaMalloc(&d_input, std::max(input.size(), size_t(1)));
        cudaMalloc(&d_output, 20);
        
        if (!input.empty()) {
            cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice);
        }
        
        LaunchSHA1Test(d_input, input.size(), d_output);
        
        uint8_t gpu_hash[20];
        cudaMemcpy(gpu_hash, d_output, 20, cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        // Compare results
        BOOST_CHECK_MESSAGE(
            memcmp(cpu_hash, gpu_hash, 20) == 0,
            "SHA1 mismatch for input: \"" + input_str + "\""
        );
    }
}

BOOST_AUTO_TEST_CASE(gpu_ripemd160_verification)
{
    // Test vectors
    std::vector<std::pair<std::string, std::string>> test_vectors = {
        {"", "9c1185a5c5e9fc54612808977ee8f548b2258d31"},
        {"abc", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"},
        {"message digest", "5d0689ef49d2fae572b881b123a85ffa21595f36"},
        {"The quick brown fox jumps over the lazy dog", "37f332f68db77bd9d7edd4969571ad671cf9dd3b"}
    };
    
    for (const auto& [input_str, expected_hex] : test_vectors) {
        std::vector<uint8_t> input(input_str.begin(), input_str.end());
        
        // CPU computation
        uint8_t cpu_hash[20];
        CRIPEMD160().Write(input.data(), input.size()).Finalize(cpu_hash);
        
        // GPU computation
        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        
        cudaMalloc(&d_input, std::max(input.size(), size_t(1)));
        cudaMalloc(&d_output, 20);
        
        if (!input.empty()) {
            cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice);
        }
        
        LaunchRIPEMD160Test(d_input, input.size(), d_output);
        
        uint8_t gpu_hash[20];
        cudaMemcpy(gpu_hash, d_output, 20, cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        // Compare results
        BOOST_CHECK_MESSAGE(
            memcmp(cpu_hash, gpu_hash, 20) == 0,
            "RIPEMD160 mismatch for input: \"" + input_str + "\""
        );
    }
}

BOOST_AUTO_TEST_CASE(gpu_hash160_verification)
{
    for (int test = 0; test < 100; test++) {
        size_t len = m_rng.randrange(200);
        std::vector<uint8_t> data(len);
        for (size_t i = 0; i < len; i++) {
            data[i] = m_rng.randbits(8);
        }
        
        // CPU computation
        uint8_t cpu_hash[20];
        CHash160().Write({data.data(), len}).Finalize({cpu_hash});
        
        // GPU computation
        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        
        cudaMalloc(&d_input, std::max(len, size_t(1)));
        cudaMalloc(&d_output, 20);
        
        if (!data.empty()) {
            cudaMemcpy(d_input, data.data(), len, cudaMemcpyHostToDevice);
        }
        
        LaunchHash160Test(d_input, len, d_output);
        
        uint8_t gpu_hash[20];
        cudaMemcpy(gpu_hash, d_output, 20, cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        BOOST_CHECK_MESSAGE(
            memcmp(cpu_hash, gpu_hash, 20) == 0,
            "Hash160 mismatch for test " + std::to_string(test)
        );
    }
}

BOOST_AUTO_TEST_CASE(gpu_siphash_verification)
{
    for (int test = 0; test < 100; test++) {
        uint64_t k0 = m_rng.rand64();
        uint64_t k1 = m_rng.rand64();
        
        size_t len = m_rng.randrange(100);
        std::vector<uint8_t> data(len);
        for (size_t i = 0; i < len; i++) {
            data[i] = m_rng.randbits(8);
        }
        
        // CPU computation
        CSipHasher cpu_hasher(k0, k1);
        cpu_hasher.Write({data.data(), len});
        uint64_t cpu_result = cpu_hasher.Finalize();
        
        // GPU computation
        uint8_t* d_data = nullptr;
        uint64_t* d_output = nullptr;
        
        cudaMalloc(&d_data, std::max(len, size_t(1)));
        cudaMalloc(&d_output, sizeof(uint64_t));
        
        if (!data.empty()) {
            cudaMemcpy(d_data, data.data(), len, cudaMemcpyHostToDevice);
        }
        
        LaunchSipHashTest(k0, k1, d_data, len, d_output);
        
        uint64_t gpu_result;
        cudaMemcpy(&gpu_result, d_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_data);
        cudaFree(d_output);
        
        BOOST_CHECK_MESSAGE(
            cpu_result == gpu_result,
            "SipHash mismatch for test " + std::to_string(test) +
            "\nExpected: " + std::to_string(cpu_result) +
            "\nGot:      " + std::to_string(gpu_result)
        );
    }
}

BOOST_AUTO_TEST_CASE(gpu_murmurhash3_verification)
{
    for (int test = 0; test < 100; test++) {
        uint32_t seed = m_rng.rand32();
        
        size_t len = m_rng.randrange(100);
        std::vector<uint8_t> data(len);
        for (size_t i = 0; i < len; i++) {
            data[i] = m_rng.randbits(8);
        }
        
        // CPU computation
        uint32_t cpu_result = MurmurHash3(seed, {data.data(), len});
        
        // GPU computation
        uint8_t* d_data = nullptr;
        uint32_t* d_output = nullptr;
        
        cudaMalloc(&d_data, std::max(len, size_t(1)));
        cudaMalloc(&d_output, sizeof(uint32_t));
        
        if (!data.empty()) {
            cudaMemcpy(d_data, data.data(), len, cudaMemcpyHostToDevice);
        }
        
        LaunchMurmurHash3Test(seed, d_data, len, d_output);
        
        uint32_t gpu_result;
        cudaMemcpy(&gpu_result, d_output, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_data);
        cudaFree(d_output);
        
        BOOST_CHECK_MESSAGE(
            cpu_result == gpu_result,
            "MurmurHash3 mismatch for test " + std::to_string(test) +
            "\nExpected: " + std::to_string(cpu_result) +
            "\nGot:      " + std::to_string(gpu_result)
        );
    }
}

BOOST_AUTO_TEST_SUITE_END()