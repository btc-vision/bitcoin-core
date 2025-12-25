// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_TX_GRAPH_H
#define BITCOIN_GPU_KERNEL_GPU_TX_GRAPH_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace gpu {

// =============================================================================
// Transaction Dependency Graph for Same-Block Batching
// =============================================================================
//
// When validating a block, transactions may depend on outputs created by
// earlier transactions in the same block. For GPU parallel validation:
//
// 1. Independent transactions can be validated in parallel
// 2. Dependent transactions must wait for their parents to complete
//
// This class builds a dependency graph and performs topological sort to
// identify parallel batches that can be sent to the GPU together.
//
// Example block with 6 transactions:
//   TX_A: No dependencies (coinbase or spends pre-block UTXOs)
//   TX_B: No dependencies
//   TX_C: Spends TX_A output
//   TX_D: Spends TX_B output
//   TX_E: Spends TX_C output (so transitively depends on TX_A)
//   TX_F: No dependencies
//
// Topological batches:
//   Batch 0: [TX_A, TX_B, TX_F]  - All independent, process in parallel
//   Batch 1: [TX_C, TX_D]        - Depend on batch 0, process in parallel
//   Batch 2: [TX_E]              - Depends on batch 1
//
// =============================================================================

// Represents a transaction input reference
struct TxInputRef {
    uint8_t prev_txid[32];  // Previous transaction hash
    uint32_t prev_vout;     // Output index in previous transaction
};

// Represents a transaction for dependency analysis
struct TxForGraph {
    uint8_t txid[32];                   // This transaction's hash
    uint32_t tx_index;                  // Index in the original block/batch
    std::vector<TxInputRef> inputs;     // All inputs this tx spends
    size_t num_outputs;                 // Number of outputs (for building UTXO set)
};

// Result of topological sort: batches of transaction indices
struct TxBatchResult {
    // Each inner vector is a batch of tx_indices that can be processed in parallel
    // Batches are ordered: batch[i] must complete before batch[i+1] can start
    std::vector<std::vector<uint32_t>> batches;

    // True if there's a cycle (invalid block - shouldn't happen with valid blocks)
    bool has_cycle{false};

    // Statistics
    size_t total_txs{0};
    size_t num_batches{0};
    size_t max_batch_size{0};
    size_t num_with_dependencies{0};  // Txs that depend on same-block txs
};

class TxDependencyGraph {
public:
    TxDependencyGraph() = default;
    ~TxDependencyGraph() = default;

    // ==========================================================================
    // Graph Building
    // ==========================================================================

    // Clear the graph for reuse
    void Clear();

    // Add a transaction to the graph
    // Returns the internal node ID for this transaction
    uint32_t AddTransaction(const TxForGraph& tx);

    // Convenience: Add transaction with raw data
    uint32_t AddTransaction(
        const uint8_t* txid,
        uint32_t tx_index,
        const TxInputRef* inputs,
        size_t num_inputs,
        size_t num_outputs
    );

    // ==========================================================================
    // Dependency Analysis
    // ==========================================================================

    // Build the dependency graph edges
    // Call this after adding all transactions
    void BuildDependencies();

    // Perform topological sort and group into parallel batches
    TxBatchResult ComputeBatches();

    // ==========================================================================
    // Query
    // ==========================================================================

    // Get number of transactions in the graph
    size_t GetTxCount() const { return m_transactions.size(); }

    // Check if a transaction depends on another transaction in the same block
    bool HasInBlockDependency(uint32_t tx_index) const;

    // Get the dependency depth (0 = no same-block deps, 1 = depends on level 0, etc.)
    uint32_t GetDependencyDepth(uint32_t tx_index) const;

private:
    // Transaction data
    std::vector<TxForGraph> m_transactions;

    // Map from txid to node index for fast lookup
    // Key: first 8 bytes of txid as uint64_t for fast comparison
    std::unordered_map<uint64_t, std::vector<uint32_t>> m_txid_to_node;

    // Adjacency list: node -> list of nodes that depend on it (children)
    std::vector<std::vector<uint32_t>> m_children;

    // Reverse adjacency: node -> list of nodes it depends on (parents)
    std::vector<std::vector<uint32_t>> m_parents;

    // Dependency depth for each node (computed during topological sort)
    std::vector<uint32_t> m_depth;

    // Helper to compute txid hash key
    static uint64_t TxidToKey(const uint8_t* txid);

    // Find node by txid (returns -1 if not found)
    int32_t FindNode(const uint8_t* txid) const;
};

// =============================================================================
// Utility Functions
// =============================================================================

// Sort transactions for optimal GPU batch processing
// Returns batches of transaction indices that can be processed in parallel
TxBatchResult SortTransactionsForGPU(
    const std::vector<TxForGraph>& transactions
);

// Convenience overload using raw transaction data
TxBatchResult SortTransactionsForGPU(
    const uint8_t* const* txids,           // Array of txid pointers (32 bytes each)
    const uint32_t* tx_indices,            // Original indices
    const TxInputRef* const* inputs,       // Array of input arrays per tx
    const size_t* num_inputs,              // Number of inputs per tx
    const size_t* num_outputs,             // Number of outputs per tx
    size_t num_transactions
);

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_TX_GRAPH_H
