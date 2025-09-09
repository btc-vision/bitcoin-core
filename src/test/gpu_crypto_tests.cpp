// Comprehensive GPU Cryptography Tests
#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <crypto/sha256.h>
#include <crypto/ripemd160.h>
#include <crypto/siphash.h>
#include <hash.h>
#include <uint256.h>
#include <random.h>

// Include GPU headers
#ifdef ENABLE_GPU_ACCELERATION
#include <gpu_kernel/gpu_crypto.cuh>
#include <gpu_kernel/gpu_hash_proper.cuh>
#include <cuda_runtime.h>
#endif

BOOST_AUTO_TEST_SUITE(gpu_crypto_tests)

#ifdef ENABLE_GPU_ACCELERATION

// Helper to run GPU hash function and copy result back
template<typename Func>
void RunGPUHash(Func func, const uint8_t* input, size_t len, uint8_t* output, size_t output_len) {
    uint8_t* d_input;
    uint8_t* d_output;
    
    cudaMalloc(&d_input, len);
    cudaMalloc(&d_output, output_len);
    
    cudaMemcpy(d_input, input, len, cudaMemcpyHostToDevice);
    
    // Run the hash function on GPU
    func<<<1, 1>>>(d_input, len, d_output);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, d_output, output_len, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

BOOST_FIXTURE_TEST_CASE(gpu_sha256_correctness, BasicTestingSetup)
{
    // Test vectors from Bitcoin Core
    struct TestVector {
        std::string input;
        std::string expected_hex;
    };
    
    TestVector vectors[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"message digest", "f7846f55cf23e14eebeab5b4e1550cad5b509e3348fbc4efa3a1413d393cb650"},
        {"abcdefghijklmnopqrstuvwxyz", "71c480df93d6ae2f1efad1447c66c9525e316218cf51fc8d9ed832f2daf18b73"},
        {"The quick brown fox jumps over the lazy dog", "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"}
    };
    
    for (const auto& test : vectors) {
        // CPU hash
        uint256 cpu_hash = Hash(test.input);
        
        // GPU hash
        uint8_t gpu_hash[32];
        RunGPUHash(
            [] __global__ (const uint8_t* data, size_t len, uint8_t* out) {
                gpu::crypto::sha256(data, len, out);
            },
            (const uint8_t*)test.input.data(),
            test.input.size(),
            gpu_hash,
            32
        );
        
        // Compare
        BOOST_CHECK_EQUAL(HexStr(Span(gpu_hash, 32)), test.expected_hex);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_sha256d_correctness, BasicTestingSetup)
{
    // Test double SHA-256 (used in Bitcoin block hashing)
    for (int i = 0; i < 100; i++) {
        // Generate random data
        std::vector<uint8_t> data(32 + (m_rng.rand32() % 100));
        for (auto& b : data) {
            b = m_rng.randbits(8);
        }
        
        // CPU double SHA-256
        CHash256 cpu_hasher;
        cpu_hasher.Write(data.data(), data.size());
        uint256 cpu_hash;
        cpu_hasher.Finalize(cpu_hash);
        
        // GPU double SHA-256
        uint8_t gpu_hash[32];
        RunGPUHash(
            [] __global__ (const uint8_t* data, size_t len, uint8_t* out) {
                gpu::crypto::sha256d(data, len, out);
            },
            data.data(),
            data.size(),
            gpu_hash,
            32
        );
        
        // Compare
        BOOST_CHECK(memcmp(cpu_hash.begin(), gpu_hash, 32) == 0);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_ripemd160_correctness, BasicTestingSetup)
{
    // Test RIPEMD-160
    struct TestVector {
        std::string input;
        std::string expected_hex;
    };
    
    TestVector vectors[] = {
        {"", "9c1185a5c5e9fc54612808977ee8f548b2258d31"},
        {"abc", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"},
        {"message digest", "5d0689ef49d2fae572b881b123a85ffa21595f36"},
        {"abcdefghijklmnopqrstuvwxyz", "f71c27109c692c1b56bbdceb5b9d2865b3708dbc"}
    };
    
    for (const auto& test : vectors) {
        // CPU RIPEMD-160
        uint160 cpu_hash = RIPEMD160(Span((const uint8_t*)test.input.data(), test.input.size()));
        
        // GPU RIPEMD-160
        uint8_t gpu_hash[20];
        RunGPUHash(
            [] __global__ (const uint8_t* data, size_t len, uint8_t* out) {
                gpu::crypto::ripemd160(data, len, out);
            },
            (const uint8_t*)test.input.data(),
            test.input.size(),
            gpu_hash,
            20
        );
        
        // Compare
        BOOST_CHECK_EQUAL(HexStr(Span(gpu_hash, 20)), test.expected_hex);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_hash160_correctness, BasicTestingSetup)
{
    // Test Hash160 (RIPEMD160(SHA256(x))) - used in Bitcoin addresses
    for (int i = 0; i < 100; i++) {
        // Generate random public key (33 bytes for compressed)
        std::vector<uint8_t> pubkey(33);
        pubkey[0] = (m_rng.randbits(1) ? 0x02 : 0x03);  // Compressed pubkey prefix
        for (int j = 1; j < 33; j++) {
            pubkey[j] = m_rng.randbits(8);
        }
        
        // CPU Hash160
        CHash160 cpu_hasher;
        cpu_hasher.Write(pubkey.data(), pubkey.size());
        uint160 cpu_hash;
        cpu_hasher.Finalize(cpu_hash);
        
        // GPU Hash160
        uint8_t gpu_hash[20];
        RunGPUHash(
            [] __global__ (const uint8_t* data, size_t len, uint8_t* out) {
                gpu::crypto::hash160(data, len, out);
            },
            pubkey.data(),
            pubkey.size(),
            gpu_hash,
            20
        );
        
        // Compare
        BOOST_CHECK(memcmp(cpu_hash.begin(), gpu_hash, 20) == 0);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_siphash_correctness, BasicTestingSetup)
{
    // Test SipHash-2-4 (used in Bitcoin Core for hash tables)
    for (int i = 0; i < 100; i++) {
        uint64_t k0 = m_rng.rand64();
        uint64_t k1 = m_rng.rand64();
        
        // Test with uint256 (common use case in Bitcoin)
        uint256 test_val = GetRandHash();
        
        // CPU SipHash
        uint64_t cpu_hash = SipHashUint256(k0, k1, test_val);
        
        // GPU SipHash
        uint64_t gpu_hash;
        uint8_t* d_data;
        uint64_t* d_result;
        
        cudaMalloc(&d_data, 32);
        cudaMalloc(&d_result, sizeof(uint64_t));
        cudaMemcpy(d_data, test_val.begin(), 32, cudaMemcpyHostToDevice);
        
        // Launch kernel
        auto kernel = [] __global__ (uint64_t k0, uint64_t k1, const uint8_t* data, uint64_t* result) {
            *result = gpu::crypto::siphash_uint256(k0, k1, data);
        };
        kernel<<<1, 1>>>(k0, k1, d_data, d_result);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&gpu_hash, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        cudaFree(d_result);
        
        BOOST_CHECK_EQUAL(cpu_hash, gpu_hash);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_murmurhash3_correctness, BasicTestingSetup)
{
    // Test MurmurHash3 (used in bloom filters)
    for (int i = 0; i < 100; i++) {
        uint32_t seed = m_rng.rand32();
        std::vector<uint8_t> data(m_rng.randrange(256));
        for (auto& b : data) {
            b = m_rng.randbits(8);
        }
        
        // CPU MurmurHash3
        uint32_t cpu_hash = MurmurHash3(seed, data);
        
        // GPU MurmurHash3
        uint32_t gpu_hash;
        uint8_t* d_data;
        uint32_t* d_result;
        
        cudaMalloc(&d_data, data.size());
        cudaMalloc(&d_result, sizeof(uint32_t));
        cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice);
        
        auto kernel = [] __global__ (uint32_t seed, const uint8_t* data, size_t len, uint32_t* result) {
            *result = gpu::crypto::murmurhash3(seed, data, len);
        };
        kernel<<<1, 1>>>(seed, d_data, data.size(), d_result);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&gpu_hash, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        cudaFree(d_result);
        
        BOOST_CHECK_EQUAL(cpu_hash, gpu_hash);
    }
}

BOOST_FIXTURE_TEST_CASE(gpu_hash_performance, BasicTestingSetup)
{
    // Performance comparison test
    const int iterations = 10000;
    std::vector<uint8_t> data(1024);
    for (auto& b : data) {
        b = m_rng.randbits(8);
    }
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        CHash256 hasher;
        hasher.Write(data.data(), data.size());
        uint256 hash;
        hasher.Finalize(hash);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    // GPU timing (launch many threads)
    uint8_t* d_data;
    uint8_t* d_results;
    cudaMalloc(&d_data, data.size());
    cudaMalloc(&d_results, iterations * 32);
    cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    auto kernel = [] __global__ (const uint8_t* data, size_t len, uint8_t* results, int count) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < count) {
            gpu::crypto::sha256d(data, len, results + tid * 32);
        }
    };
    
    int blockSize = 256;
    int gridSize = (iterations + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(d_data, data.size(), d_results, iterations);
    cudaDeviceSynchronize();
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    
    cudaFree(d_data);
    cudaFree(d_results);
    
    // GPU should be significantly faster for parallel operations
    double speedup = (double)cpu_time / gpu_time;
    BOOST_TEST_MESSAGE("SHA256D Performance - CPU: " << cpu_time << "μs, GPU: " << gpu_time << "μs, Speedup: " << speedup << "x");
    
    // GPU should be at least 10x faster for this parallel workload
    BOOST_CHECK_GT(speedup, 10.0);
}

#else

BOOST_AUTO_TEST_CASE(gpu_crypto_disabled)
{
    BOOST_TEST_MESSAGE("GPU crypto tests skipped - GPU acceleration not enabled");
}

#endif // ENABLE_GPU_ACCELERATION

BOOST_AUTO_TEST_SUITE_END()