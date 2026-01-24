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
#include <vector>

using namespace kuzu::binder;
using namespace kuzu::common;
using namespace kuzu::processor;
using namespace kuzu::storage;
using namespace kuzu::graph;
using namespace kuzu::function;

namespace kuzu {
namespace algo_extension {

static constexpr char CLOSENESS_COLUMN_NAME[] = "closeness";

// Bind data for closeness centrality - uses GDS pattern for per-node output
struct ClosenessBindData final : public GDSBindData {
    ClosenessBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<ClosenessBindData>(*this);
    }
};

// Stores per-node closeness centrality values
class ClosenessValues {
public:
    ClosenessValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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

private:
    std::atomic<double>* values = nullptr;
    GDSDenseObjectManager<std::atomic<double>> valueMap;
};

// Output vertex compute class - writes results to factorized table
class ClosenessResultVertexCompute : public GDSResultVertexCompute {
public:
    ClosenessResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        ClosenessValues& closenessValues)
        : GDSResultVertexCompute{mm, sharedState}, closenessValues{closenessValues} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        closenessVector = createVector(LogicalType::DOUBLE());
    }

    void beginOnTableInternal(table_id_t tableID) override { closenessValues.pinTable(tableID); }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            closenessVector->setValue<double>(0, closenessValues.getValue(i));
            localFT->append(vectors);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<ClosenessResultVertexCompute>(mm, sharedState, closenessValues);
    }

private:
    ClosenessValues& closenessValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> closenessVector;
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

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    auto clientContext = input.context->clientContext;
    auto transaction = transaction::Transaction::Get(*clientContext);
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto graph = sharedState->graph.get();
    auto maxOffsetMap = graph->getMaxOffsetMap(transaction);
    auto mm = MemoryManager::Get(*clientContext);

    // Count total nodes
    uint64_t totalNodes = 0;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
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

    // Step 2: Initialize closeness values
    auto closenessValues = ClosenessValues(maxOffsetMap, mm);

    // Step 3: Compute closeness centrality for each source node using BFS
    for (const auto& [sourceTableID, maxOffset] : maxOffsetMap) {
        for (offset_t sourceOffset = 0; sourceOffset < maxOffset; ++sourceOffset) {
            // Distance map: tableID -> offset -> distance (-1 = unvisited)
            table_id_map_t<std::vector<int64_t>> distances;
            for (const auto& [tableID, tableMaxOffset] : maxOffsetMap) {
                distances[tableID].resize(tableMaxOffset, -1);
            }

            // Set source distance to 0
            distances[sourceTableID][sourceOffset] = 0;

            // BFS queue
            std::queue<nodeID_t> bfsQueue;
            bfsQueue.push(nodeID_t{sourceOffset, sourceTableID});

            while (!bfsQueue.empty()) {
                auto current = bfsQueue.front();
                bfsQueue.pop();

                auto currentDist = distances[current.tableID][current.offset];

                for (const auto& neighbor : adjLists.getNeighbors(current.tableID, current.offset)) {
                    if (distances[neighbor.tableID][neighbor.offset] == -1) {
                        distances[neighbor.tableID][neighbor.offset] = currentDist + 1;
                        bfsQueue.push(neighbor);
                    }
                }
            }

            // Sum up distances and count reachable nodes
            int64_t sumDistances = 0;
            uint64_t reachableCount = 0;

            for (const auto& [tableID, distVec] : distances) {
                for (const auto& dist : distVec) {
                    if (dist > 0) {  // Don't count self (distance 0) or unreachable (-1)
                        sumDistances += dist;
                        reachableCount++;
                    }
                }
            }

            // Calculate closeness centrality using NetworkX formula:
            // closeness(u) = (n - 1) / sum(d(u, v))
            // For disconnected graphs (Wasserman & Faust normalization):
            // closeness(u) = ((r - 1) / (n - 1)) * ((r - 1) / sum(d(u, v)))
            // where r = number of reachable nodes (including self)
            double closeness = 0.0;
            if (sumDistances > 0) {
                // r = reachableCount + 1 (reachable includes only dist > 0, so add 1 for self)
                uint64_t r = reachableCount + 1;
                if (r == totalNodes) {
                    // Connected graph: simple formula
                    closeness = static_cast<double>(totalNodes - 1) / static_cast<double>(sumDistances);
                } else {
                    // Disconnected graph: Wasserman & Faust normalization
                    double rMinus1 = static_cast<double>(r - 1);
                    double nMinus1 = static_cast<double>(totalNodes - 1);
                    closeness = (rMinus1 / nMinus1) * (rMinus1 / static_cast<double>(sumDistances));
                }
            }

            closenessValues.pinTable(sourceTableID);
            closenessValues.setValue(sourceOffset, closeness);
        }
    }

    // Step 4: Output results
    auto outputVC = std::make_unique<ClosenessResultVertexCompute>(mm, sharedState, closenessValues);
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
        input->binder->createVariable(CLOSENESS_COLUMN_NAME, LogicalType::DOUBLE()));
    return std::make_unique<ClosenessBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput);
}

function_set ClosenessCentralityFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(ClosenessCentralityFunction::name,
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
