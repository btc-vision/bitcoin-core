// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "gpu_tx_graph.h"
#include "gpu_logging.h"
#include <algorithm>
#include <queue>
#include <cstring>

namespace gpu {

// =============================================================================
// TxDependencyGraph Implementation
// =============================================================================

void TxDependencyGraph::Clear() {
    m_transactions.clear();
    m_txid_to_node.clear();
    m_children.clear();
    m_parents.clear();
    m_depth.clear();
}

uint64_t TxDependencyGraph::TxidToKey(const uint8_t* txid) {
    // Use first 8 bytes of txid as hash key
    // This is sufficient for collision resistance within a single block
    uint64_t key = 0;
    for (int i = 0; i < 8; i++) {
        key |= static_cast<uint64_t>(txid[i]) << (i * 8);
    }
    return key;
}

int32_t TxDependencyGraph::FindNode(const uint8_t* txid) const {
    uint64_t key = TxidToKey(txid);
    auto it = m_txid_to_node.find(key);
    if (it == m_txid_to_node.end()) {
        return -1;
    }

    // Check all candidates with this key (handle potential collisions)
    for (uint32_t node_idx : it->second) {
        if (memcmp(m_transactions[node_idx].txid, txid, 32) == 0) {
            return static_cast<int32_t>(node_idx);
        }
    }

    return -1;
}

uint32_t TxDependencyGraph::AddTransaction(const TxForGraph& tx) {
    uint32_t node_idx = static_cast<uint32_t>(m_transactions.size());
    m_transactions.push_back(tx);

    // Add to txid lookup map
    uint64_t key = TxidToKey(tx.txid);
    m_txid_to_node[key].push_back(node_idx);

    return node_idx;
}

uint32_t TxDependencyGraph::AddTransaction(
    const uint8_t* txid,
    uint32_t tx_index,
    const TxInputRef* inputs,
    size_t num_inputs,
    size_t num_outputs)
{
    TxForGraph tx;
    memcpy(tx.txid, txid, 32);
    tx.tx_index = tx_index;
    tx.num_outputs = num_outputs;

    if (inputs && num_inputs > 0) {
        tx.inputs.assign(inputs, inputs + num_inputs);
    }

    return AddTransaction(tx);
}

void TxDependencyGraph::BuildDependencies() {
    size_t n = m_transactions.size();
    m_children.resize(n);
    m_parents.resize(n);
    m_depth.resize(n, 0);

    // Clear any existing edges
    for (size_t i = 0; i < n; i++) {
        m_children[i].clear();
        m_parents[i].clear();
    }

    // For each transaction, check if any of its inputs spend outputs from
    // another transaction in this block
    for (size_t i = 0; i < n; i++) {
        const TxForGraph& tx = m_transactions[i];

        for (const TxInputRef& input : tx.inputs) {
            int32_t parent_node = FindNode(input.prev_txid);

            if (parent_node >= 0 && parent_node != static_cast<int32_t>(i)) {
                // This transaction depends on another transaction in the same block
                // parent_node must be processed before node i
                m_children[parent_node].push_back(static_cast<uint32_t>(i));
                m_parents[i].push_back(static_cast<uint32_t>(parent_node));
            }
        }
    }

    // Remove duplicate edges (can happen if tx spends multiple outputs from same parent)
    for (size_t i = 0; i < n; i++) {
        std::sort(m_children[i].begin(), m_children[i].end());
        m_children[i].erase(std::unique(m_children[i].begin(), m_children[i].end()), m_children[i].end());

        std::sort(m_parents[i].begin(), m_parents[i].end());
        m_parents[i].erase(std::unique(m_parents[i].begin(), m_parents[i].end()), m_parents[i].end());
    }
}

TxBatchResult TxDependencyGraph::ComputeBatches() {
    TxBatchResult result;
    result.total_txs = m_transactions.size();

    if (m_transactions.empty()) {
        return result;
    }

    size_t n = m_transactions.size();

    // Kahn's algorithm for topological sort with level tracking
    // Level = the batch number this transaction belongs to

    // In-degree count for each node
    std::vector<uint32_t> in_degree(n, 0);
    for (size_t i = 0; i < n; i++) {
        in_degree[i] = static_cast<uint32_t>(m_parents[i].size());
        if (in_degree[i] > 0) {
            result.num_with_dependencies++;
        }
    }

    // Queue of (node, level) pairs
    std::queue<std::pair<uint32_t, uint32_t>> queue;

    // Start with all nodes that have no parents (in-degree = 0)
    for (size_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            queue.push({static_cast<uint32_t>(i), 0});
            m_depth[i] = 0;
        }
    }

    // Group nodes by their level (batch)
    std::unordered_map<uint32_t, std::vector<uint32_t>> level_to_nodes;
    uint32_t max_level = 0;
    size_t processed = 0;

    while (!queue.empty()) {
        auto [node, level] = queue.front();
        queue.pop();

        level_to_nodes[level].push_back(m_transactions[node].tx_index);
        max_level = std::max(max_level, level);
        processed++;

        // Process all children
        for (uint32_t child : m_children[node]) {
            in_degree[child]--;
            if (in_degree[child] == 0) {
                // Child's level is max of all its parents' levels + 1
                uint32_t child_level = 0;
                for (uint32_t parent : m_parents[child]) {
                    child_level = std::max(child_level, m_depth[parent] + 1);
                }
                m_depth[child] = child_level;
                queue.push({child, child_level});
            }
        }
    }

    // Check for cycle (shouldn't happen with valid blocks)
    if (processed != n) {
        result.has_cycle = true;
        LogGPUInfo("Warning: Cycle detected in transaction dependencies!");

        // Still try to return partial results - include unprocessed nodes in a final batch
        std::vector<uint32_t> unprocessed;
        for (size_t i = 0; i < n; i++) {
            if (in_degree[i] > 0) {
                unprocessed.push_back(m_transactions[i].tx_index);
            }
        }
        if (!unprocessed.empty()) {
            level_to_nodes[max_level + 1] = unprocessed;
            max_level++;
        }
    }

    // Build result batches in order
    result.batches.resize(max_level + 1);
    for (uint32_t level = 0; level <= max_level; level++) {
        result.batches[level] = std::move(level_to_nodes[level]);
        result.max_batch_size = std::max(result.max_batch_size, result.batches[level].size());
    }
    result.num_batches = result.batches.size();

    return result;
}

bool TxDependencyGraph::HasInBlockDependency(uint32_t tx_index) const {
    // Find the node for this tx_index
    for (size_t i = 0; i < m_transactions.size(); i++) {
        if (m_transactions[i].tx_index == tx_index) {
            return !m_parents[i].empty();
        }
    }
    return false;
}

uint32_t TxDependencyGraph::GetDependencyDepth(uint32_t tx_index) const {
    for (size_t i = 0; i < m_transactions.size(); i++) {
        if (m_transactions[i].tx_index == tx_index) {
            return m_depth[i];
        }
    }
    return 0;
}

// =============================================================================
// Utility Functions
// =============================================================================

TxBatchResult SortTransactionsForGPU(const std::vector<TxForGraph>& transactions) {
    TxDependencyGraph graph;

    for (const auto& tx : transactions) {
        graph.AddTransaction(tx);
    }

    graph.BuildDependencies();
    return graph.ComputeBatches();
}

TxBatchResult SortTransactionsForGPU(
    const uint8_t* const* txids,
    const uint32_t* tx_indices,
    const TxInputRef* const* inputs,
    const size_t* num_inputs,
    const size_t* num_outputs,
    size_t num_transactions)
{
    TxDependencyGraph graph;

    for (size_t i = 0; i < num_transactions; i++) {
        graph.AddTransaction(
            txids[i],
            tx_indices[i],
            inputs[i],
            num_inputs[i],
            num_outputs[i]
        );
    }

    graph.BuildDependencies();
    return graph.ComputeBatches();
}

} // namespace gpu
