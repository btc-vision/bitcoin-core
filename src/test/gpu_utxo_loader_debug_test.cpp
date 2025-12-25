// Debug test for GPU UTXO loader
#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <coins.h>
#include <txdb.h>
#include <uint256.h>
#include <primitives/transaction.h>
#include <script/script.h>
#include <random.h>
#include <iostream>

BOOST_FIXTURE_TEST_SUITE(gpu_utxo_loader_debug, TestingSetup)

BOOST_AUTO_TEST_CASE(test_view_hierarchy)
{
    // Create an in-memory database
    DBParams db_params{
        .path = "",  // Empty path for in-memory
        .cache_bytes = 10 << 20,  // 10MB
        .memory_only = true,
        .wipe_data = true,
    };
    CoinsViewOptions options;
    
    std::cout << "Creating CCoinsViewDB..." << std::endl;
    auto db = std::make_unique<CCoinsViewDB>(db_params, options);
    
    std::cout << "Creating CCoinsViewCache..." << std::endl;
    CCoinsViewCache cache(db.get());
    
    // Set a best block
    FastRandomContext rng(42);
    uint256 bestBlock = rng.rand256();
    cache.SetBestBlock(bestBlock);
    
    std::cout << "Testing view hierarchy navigation..." << std::endl;
    
    // Test getting base
    CCoinsView* base1 = cache.GetBase();
    std::cout << "cache.GetBase() = " << (void*)base1 << std::endl;
    std::cout << "Expected db.get() = " << (void*)db.get() << std::endl;
    
    // Test cursor from database directly
    std::cout << "Getting cursor from CCoinsViewDB directly..." << std::endl;
    auto cursor1 = db->Cursor();
    std::cout << "Direct DB cursor = " << (cursor1 ? "SUCCESS" : "NULL") << std::endl;
    
    // Test cursor through base
    std::cout << "Getting cursor through base pointer..." << std::endl;
    auto cursor2 = base1->Cursor();
    std::cout << "Base cursor = " << (cursor2 ? "SUCCESS" : "NULL") << std::endl;
    
    // Add a coin and flush
    COutPoint outpoint(Txid::FromUint256(rng.rand256()), 0);
    CTxOut output(100000000, CScript() << OP_TRUE);
    Coin coin(std::move(output), 100, false);
    
    cache.AddCoin(outpoint, std::move(coin), false);
    
    std::cout << "Flushing cache..." << std::endl;
    bool flushResult = cache.Flush();
    std::cout << "Flush result = " << (flushResult ? "SUCCESS" : "FAILED") << std::endl;
    
    // Try cursor again after flush
    std::cout << "Getting cursor after flush..." << std::endl;
    auto cursor3 = db->Cursor();
    std::cout << "Post-flush cursor = " << (cursor3 ? "SUCCESS" : "NULL") << std::endl;
    
    if (cursor3) {
        std::cout << "Cursor is valid: " << cursor3->Valid() << std::endl;
        if (cursor3->Valid()) {
            COutPoint key;
            Coin value;
            if (cursor3->GetKey(key)) {
                std::cout << "Got key from cursor!" << std::endl;
            }
            if (cursor3->GetValue(value)) {
                std::cout << "Got value from cursor!" << std::endl;
            }
        }
    }
    
    BOOST_CHECK(true); // Dummy check
}

BOOST_AUTO_TEST_SUITE_END()