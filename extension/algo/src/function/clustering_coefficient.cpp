#include "binder/binder.h"
#include "function/algo_function.h"
#include "function/degrees.h"
#include "function/gds/gds.h"
#include "function/gds/gds_utils.h"
#include "function/gds/gds_vertex_compute.h"
#include "function/table/bind_data.h"
#include "function/table/bind_input.h"
#include "graph/on_disk_graph.h"
#include "processor/execution_context.h"
#include "transaction/transaction.h"

#include <algorithm>
#include <vector>

using namespace kuzu::binder;
using namespace kuzu::common;
using namespace kuzu::processor;
using namespace kuzu::storage;
using namespace kuzu::graph;
using namespace kuzu::function;

namespace kuzu {
namespace algo_extension {

static constexpr char COEFFICIENT_COLUMN_NAME[] = "coefficient";

// Bind data for clustering coefficient - uses GDS pattern for per-node output
struct ClusteringCoefficientBindData final : public GDSBindData {
    ClusteringCoefficientBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<ClusteringCoefficientBindData>(*this);
    }
};

// Stores per-node clustering coefficient values
class CCValues {
public:
    CCValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class CCResultVertexCompute : public GDSResultVertexCompute {
public:
    CCResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        CCValues& ccValues)
        : GDSResultVertexCompute{mm, sharedState}, ccValues{ccValues} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        coefficientVector = createVector(LogicalType::DOUBLE());
    }

    void beginOnTableInternal(table_id_t tableID) override { ccValues.pinTable(tableID); }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            coefficientVector->setValue<double>(0, ccValues.getValue(i));
            localFT->append(vectors);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<CCResultVertexCompute>(mm, sharedState, ccValues);
    }

private:
    CCValues& ccValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> coefficientVector;
};

// Adjacency list structure - stores neighbors as sorted vectors for binary search
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
        // Sort all adjacency lists for binary search
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

    bool hasEdge(nodeID_t from, nodeID_t to) const {
        const auto& neighbors = adjacencyMap.at(from.tableID)[from.offset];
        return std::binary_search(neighbors.begin(), neighbors.end(), to,
            [](const nodeID_t& a, const nodeID_t& b) {
                if (a.tableID != b.tableID) {
                    return a.tableID < b.tableID;
                }
                return a.offset < b.offset;
            });
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

    // Step 2: Compute degrees from adjacency lists
    auto degrees = Degrees(maxOffsetMap, mm);
    DegreesUtils::computeDegree(input.context, graph, sharedState->getGraphNodeMaskMap(), &degrees,
        ExtendDirection::BOTH);

    // Step 3: Initialize CC values and compute clustering coefficient for each node
    auto ccValues = CCValues(maxOffsetMap, mm);

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        degrees.pinTable(tableID);
        ccValues.pinTable(tableID);

        for (offset_t i = 0; i < maxOffset; ++i) {
            auto degree = degrees.getValue(i);
            // Nodes with degree < 2 have undefined CC (set to 0)
            if (degree < 2) {
                ccValues.setValue(i, 0.0);
                continue;
            }

            // Count triangles: for each pair of neighbors, check if they're connected
            const auto& neighbors = adjLists.getNeighbors(tableID, i);
            uint64_t triangles = 0;
            for (size_t j = 0; j < neighbors.size(); ++j) {
                for (size_t k = j + 1; k < neighbors.size(); ++k) {
                    if (adjLists.hasEdge(neighbors[j], neighbors[k])) {
                        triangles++;
                    }
                }
            }

            // CC = (2 * triangles) / (degree * (degree - 1))
            // Note: degree from Degrees class counts both directions, but neighbors list is unique
            // So we use neighbors.size() for the actual neighbor count
            auto neighborCount = neighbors.size();
            double cc = 0.0;
            if (neighborCount >= 2) {
                cc = (2.0 * triangles) /
                     (static_cast<double>(neighborCount) * static_cast<double>(neighborCount - 1));
            }
            ccValues.setValue(i, cc);
        }
    }

    // Step 4: Output results
    auto outputVC = std::make_unique<CCResultVertexCompute>(mm, sharedState, ccValues);
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
        input->binder->createVariable(COEFFICIENT_COLUMN_NAME, LogicalType::DOUBLE()));
    return std::make_unique<ClusteringCoefficientBindData>(std::move(columns),
        std::move(graphEntry), nodeOutput);
}

function_set ClusteringCoefficientFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(ClusteringCoefficientFunction::name,
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
