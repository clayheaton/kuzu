#include "binder/binder.h"
#include "function/algo_function.h"
#include "function/gds/gds.h"
#include "function/gds/gds_utils.h"
#include "function/gds/gds_vertex_compute.h"
#include "function/table/bind_data.h"
#include "function/table/bind_input.h"
#include "graph/on_disk_graph.h"
#include "processor/execution_context.h"
#include "transaction/transaction.h"

#include <algorithm>
#include <queue>
#include <stack>
#include <vector>

using namespace kuzu::binder;
using namespace kuzu::common;
using namespace kuzu::processor;
using namespace kuzu::storage;
using namespace kuzu::graph;
using namespace kuzu::function;

namespace kuzu {
namespace algo_extension {

static constexpr char BETWEENNESS_COLUMN_NAME[] = "betweenness";

// Bind data for betweenness centrality - uses GDS pattern for per-node output
struct BetweennessBindData final : public GDSBindData {
    BetweennessBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<BetweennessBindData>(*this);
    }
};

// Stores per-node betweenness centrality values
class BetweennessValues {
public:
    BetweennessValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
        for (const auto& [tableID, maxOffset] : maxOffsetMap) {
            valueMap.allocate(tableID, maxOffset, mm);
            pinTable(tableID);
            for (auto i = 0u; i < maxOffset; ++i) {
                values[i].store(0.0, std::memory_order_relaxed);
            }
        }
    }

    void pinTable(table_id_t tableID) { values = valueMap.getData(tableID); }

    double getValue(offset_t offset) { return values[offset].load(std::memory_order_relaxed); }

    void setValue(offset_t offset, double val) {
        values[offset].store(val, std::memory_order_relaxed);
    }

    void addValue(offset_t offset, double val) {
        // Non-atomic add since we're single-threaded
        auto current = values[offset].load(std::memory_order_relaxed);
        values[offset].store(current + val, std::memory_order_relaxed);
    }

private:
    std::atomic<double>* values = nullptr;
    GDSDenseObjectManager<std::atomic<double>> valueMap;
};

// Output vertex compute class - writes results to factorized table
class BetweennessResultVertexCompute : public GDSResultVertexCompute {
public:
    BetweennessResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        BetweennessValues& betweennessValues)
        : GDSResultVertexCompute{mm, sharedState}, betweennessValues{betweennessValues} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        betweennessVector = createVector(LogicalType::DOUBLE());
    }

    void beginOnTableInternal(table_id_t tableID) override { betweennessValues.pinTable(tableID); }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            betweennessVector->setValue<double>(0, betweennessValues.getValue(i));
            localFT->append(vectors);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<BetweennessResultVertexCompute>(mm, sharedState, betweennessValues);
    }

private:
    BetweennessValues& betweennessValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> betweennessVector;
};

// Adjacency list structure - stores neighbors as sorted vectors
class AdjacencyLists {
public:
    AdjacencyLists(table_id_map_t<offset_t> maxOffsetMap) {
        for (const auto& [tableID, maxOffset] : maxOffsetMap) {
            adjacencyMap[tableID].resize(maxOffset);
        }
    }

    void addNeighbor(table_id_t tableID, offset_t nodeOffset, nodeID_t neighbor) {
        adjacencyMap[tableID][nodeOffset].push_back(neighbor);
    }

    void finalize() {
        // Sort all adjacency lists and remove duplicates
        for (auto& [tableID, lists] : adjacencyMap) {
            for (auto& neighbors : lists) {
                std::sort(neighbors.begin(), neighbors.end(),
                    [](const nodeID_t& a, const nodeID_t& b) {
                        if (a.tableID != b.tableID) {
                            return a.tableID < b.tableID;
                        }
                        return a.offset < b.offset;
                    });
                // Remove duplicates (for undirected we might add same edge twice)
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end(),
                                    [](const nodeID_t& a, const nodeID_t& b) {
                                        return a.tableID == b.tableID && a.offset == b.offset;
                                    }),
                    neighbors.end());
            }
        }
    }

    const std::vector<nodeID_t>& getNeighbors(table_id_t tableID, offset_t offset) const {
        return adjacencyMap.at(tableID)[offset];
    }

private:
    table_id_map_t<std::vector<std::vector<nodeID_t>>> adjacencyMap;
};

// Edge compute to build adjacency lists
class BuildAdjacencyEdgeCompute : public EdgeCompute {
public:
    explicit BuildAdjacencyEdgeCompute(AdjacencyLists* adjLists) : adjLists{adjLists} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, NbrScanState::Chunk& chunk,
        bool) override {
        chunk.forEach([&](auto nbrNodes, auto, auto i) {
            adjLists->addNeighbor(boundNodeID.tableID, boundNodeID.offset, nbrNodes[i]);
        });
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<BuildAdjacencyEdgeCompute>(adjLists);
    }

private:
    AdjacencyLists* adjLists;
};

class BuildAdjacencyAuxiliaryState : public GDSAuxiliaryState {
public:
    void beginFrontierCompute(table_id_t, table_id_t) override {}
    void switchToDense(ExecutionContext*, Graph*) override {}
};

// Helper to convert nodeID to a linear index for local arrays
inline uint64_t nodeToIndex(nodeID_t node, const table_id_map_t<uint64_t>& tableOffsets) {
    return tableOffsets.at(node.tableID) + node.offset;
}

// Helper to convert linear index back to nodeID
inline nodeID_t indexToNode(uint64_t index, const std::vector<std::pair<table_id_t, offset_t>>& tableRanges) {
    for (const auto& [tableID, maxOffset] : tableRanges) {
        if (index < maxOffset) {
            return nodeID_t{static_cast<offset_t>(index), tableID};
        }
        index -= maxOffset;
    }
    // Should never reach here
    return nodeID_t{0, 0};
}

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    auto clientContext = input.context->clientContext;
    auto transaction = transaction::Transaction::Get(*clientContext);
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto graph = sharedState->graph.get();
    auto maxOffsetMap = graph->getMaxOffsetMap(transaction);
    auto mm = MemoryManager::Get(*clientContext);

    // Count total nodes and build table offset map for linear indexing
    uint64_t totalNodes = 0;
    table_id_map_t<uint64_t> tableOffsets;
    std::vector<std::pair<table_id_t, offset_t>> tableRanges;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        tableOffsets[tableID] = totalNodes;
        tableRanges.emplace_back(tableID, maxOffset);
        totalNodes += maxOffset;
    }

    // Step 1: Build adjacency lists by scanning all edges
    auto adjLists = AdjacencyLists(maxOffsetMap);
    {
        auto currentFrontier = DenseFrontier::getUnvisitedFrontier(input.context, graph);
        auto nextFrontier =
            DenseFrontier::getVisitedFrontier(input.context, graph, sharedState->getGraphNodeMaskMap());
        auto frontierPair =
            std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
        frontierPair->setActiveNodesForNextIter();
        auto edgeCompute = std::make_unique<BuildAdjacencyEdgeCompute>(&adjLists);
        auto auxiliaryState = std::make_unique<BuildAdjacencyAuxiliaryState>();
        auto computeState =
            GDSComputeState(std::move(frontierPair), std::move(edgeCompute), std::move(auxiliaryState));
        // Scan in BOTH directions to get all neighbors (undirected treatment)
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1 /* maxIters */);
    }
    adjLists.finalize();

    // Step 2: Initialize betweenness values to 0
    auto betweennessValues = BetweennessValues(maxOffsetMap, mm);

    // Step 3: Brandes algorithm - compute betweenness centrality
    // For each source node, run BFS and backward dependency accumulation
    for (const auto& [sourceTableID, maxOffset] : maxOffsetMap) {
        for (offset_t sourceOffset = 0; sourceOffset < maxOffset; ++sourceOffset) {
            nodeID_t source{sourceOffset, sourceTableID};

            // Data structures for this source
            // d[v] = distance from source (-1 = unvisited)
            std::vector<int64_t> d(totalNodes, -1);
            // sigma[v] = number of shortest paths from source to v
            std::vector<double> sigma(totalNodes, 0.0);
            // P[v] = predecessors of v on shortest paths from source
            std::vector<std::vector<uint64_t>> P(totalNodes);
            // S = stack of nodes in order of discovery (for backward pass)
            std::stack<uint64_t> S;
            // delta[v] = dependency of source on v
            std::vector<double> delta(totalNodes, 0.0);

            // Initialize source
            uint64_t sourceIdx = nodeToIndex(source, tableOffsets);
            d[sourceIdx] = 0;
            sigma[sourceIdx] = 1.0;

            // BFS queue
            std::queue<uint64_t> Q;
            Q.push(sourceIdx);

            // Forward BFS phase
            while (!Q.empty()) {
                uint64_t vIdx = Q.front();
                Q.pop();
                S.push(vIdx);

                nodeID_t v = indexToNode(vIdx, tableRanges);

                for (const auto& neighbor : adjLists.getNeighbors(v.tableID, v.offset)) {
                    uint64_t wIdx = nodeToIndex(neighbor, tableOffsets);

                    // First visit to w?
                    if (d[wIdx] < 0) {
                        d[wIdx] = d[vIdx] + 1;
                        Q.push(wIdx);
                    }

                    // Is edge (v, w) on a shortest path?
                    if (d[wIdx] == d[vIdx] + 1) {
                        sigma[wIdx] += sigma[vIdx];
                        P[wIdx].push_back(vIdx);
                    }
                }
            }

            // Backward pass - accumulate dependencies
            while (!S.empty()) {
                uint64_t wIdx = S.top();
                S.pop();

                for (uint64_t vIdx : P[wIdx]) {
                    delta[vIdx] += (sigma[vIdx] / sigma[wIdx]) * (1.0 + delta[wIdx]);
                }

                // Add to betweenness (skip source)
                if (wIdx != sourceIdx) {
                    nodeID_t w = indexToNode(wIdx, tableRanges);
                    betweennessValues.pinTable(w.tableID);
                    betweennessValues.addValue(w.offset, delta[wIdx]);
                }
            }
        }
    }

    // Step 4: Normalize - divide by 2 for undirected graphs (each path counted twice)
    // Then normalize by (n-1)*(n-2) for comparability with NetworkX normalized=True
    double normFactor = 1.0;
    if (totalNodes > 2) {
        // Undirected normalization: divide by 2, then by (n-1)*(n-2)
        // Combined: divide by 2 * (n-1) * (n-2) / 2 = (n-1) * (n-2)
        // Wait, NetworkX normalized divides by (n-1)*(n-2)/2 for undirected
        // And the raw undirected BC is already divided by 2
        // So: raw_bc / 2 / ((n-1)*(n-2)/2) = raw_bc / ((n-1)*(n-2))
        normFactor = static_cast<double>((totalNodes - 1) * (totalNodes - 2));
    }

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        betweennessValues.pinTable(tableID);
        for (offset_t i = 0; i < maxOffset; ++i) {
            double val = betweennessValues.getValue(i);
            // Divide by 2 for undirected, then normalize
            if (normFactor > 0) {
                betweennessValues.setValue(i, val / normFactor);
            }
        }
    }

    // Step 5: Output results
    auto outputVC = std::make_unique<BetweennessResultVertexCompute>(mm, sharedState, betweennessValues);
    GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
    sharedState->factorizedTablePool.mergeLocalTables();
    return 0;
}

static std::unique_ptr<TableFuncBindData> bindFunc(main::ClientContext* context,
    const TableFuncBindInput* input) {
    auto graphName = input->getLiteralVal<std::string>(0);
    auto graphEntry = GDSFunction::bindGraphEntry(*context, graphName);
    auto nodeOutput = GDSFunction::bindNodeOutput(*input, graphEntry.getNodeEntries());
    expression_vector columns;
    columns.push_back(nodeOutput->constCast<NodeExpression>().getInternalID());
    columns.push_back(
        input->binder->createVariable(BETWEENNESS_COLUMN_NAME, LogicalType::DOUBLE()));
    return std::make_unique<BetweennessBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput);
}

function_set BetweennessCentralityFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(BetweennessCentralityFunction::name,
        std::vector<LogicalTypeID>{LogicalTypeID::ANY});
    func->bindFunc = bindFunc;
    func->tableFunc = tableFunc;
    func->initSharedStateFunc = GDSFunction::initSharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->canParallelFunc = [] { return false; };
    func->getLogicalPlanFunc = GDSFunction::getLogicalPlan;
    func->getPhysicalPlanFunc = GDSFunction::getPhysicalPlan;
    result.push_back(std::move(func));
    return result;
}

} // namespace algo_extension
} // namespace kuzu
