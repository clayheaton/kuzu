#include "binder/binder.h"
#include "common/string_utils.h"
#include "function/algo_function.h"
#include "function/config/radial_config.h"
#include "function/degrees.h"
#include "function/gds/gds.h"
#include "function/gds/gds_utils.h"
#include "function/gds/gds_vertex_compute.h"
#include "function/table/bind_data.h"
#include "function/table/bind_input.h"
#include "function/table/optional_params.h"
#include "graph/on_disk_graph.h"
#include "processor/execution_context.h"
#include "transaction/transaction.h"

#include <algorithm>
#include <cmath>
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

static constexpr char X_COLUMN_NAME[] = "x";
static constexpr char Y_COLUMN_NAME[] = "y";

// M_PI may not be defined on all platforms
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Optional parameters for Radial layout
struct RadialOptionalParams final : public function::OptionalParams {
    function::OptionalParam<RadialMinRadius> minRadius;
    function::OptionalParam<RadialRadiusIncrement> radiusIncrement;

    explicit RadialOptionalParams(const binder::expression_vector& optionalParams) {
        for (auto& optionalParam : optionalParams) {
            auto paramName = StringUtils::getLower(optionalParam->getAlias());
            if (paramName == RadialMinRadius::NAME) {
                minRadius = function::OptionalParam<RadialMinRadius>(optionalParam);
            } else if (paramName == RadialRadiusIncrement::NAME) {
                radiusIncrement = function::OptionalParam<RadialRadiusIncrement>(optionalParam);
            } else {
                throw BinderException{"Unknown optional parameter: " + optionalParam->getAlias()};
            }
        }
    }

    // For copy only
    RadialOptionalParams(function::OptionalParam<RadialMinRadius> minRadius,
        function::OptionalParam<RadialRadiusIncrement> radiusIncrement)
        : minRadius{std::move(minRadius)}, radiusIncrement{std::move(radiusIncrement)} {}

    void evaluateParams(main::ClientContext* context) override {
        minRadius.evaluateParam(context);
        radiusIncrement.evaluateParam(context);
    }

    std::unique_ptr<function::OptionalParams> copy() override {
        return std::make_unique<RadialOptionalParams>(minRadius, radiusIncrement);
    }
};

// Bind data for radial layout
struct RadialLayoutBindData final : public GDSBindData {
    RadialLayoutBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<RadialOptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {
        this->optionalParams = std::move(optionalParams);
    }

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<RadialLayoutBindData>(*this);
    }
};

// Stores per-node X coordinate values
class RadialLayoutXValues {
public:
    RadialLayoutXValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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

// Stores per-node Y coordinate values
class RadialLayoutYValues {
public:
    RadialLayoutYValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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

// Output vertex compute class
class RadialLayoutResultVertexCompute : public GDSResultVertexCompute {
public:
    RadialLayoutResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        RadialLayoutXValues& xValues, RadialLayoutYValues& yValues)
        : GDSResultVertexCompute{mm, sharedState}, xValues{xValues}, yValues{yValues} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        xVector = createVector(LogicalType::DOUBLE());
        yVector = createVector(LogicalType::DOUBLE());
    }

    void beginOnTableInternal(table_id_t tableID) override {
        xValues.pinTable(tableID);
        yValues.pinTable(tableID);
    }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            xVector->setValue<double>(0, xValues.getValue(i));
            yVector->setValue<double>(0, yValues.getValue(i));
            localFT->append(vectors);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<RadialLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
    }

private:
    RadialLayoutXValues& xValues;
    RadialLayoutYValues& yValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> xVector;
    std::unique_ptr<ValueVector> yVector;
};

// Adjacency list for BFS
class RadialAdjacencyLists {
public:
    RadialAdjacencyLists(table_id_map_t<offset_t> maxOffsetMap) {
        for (const auto& [tableID, maxOffset] : maxOffsetMap) {
            adjacencyMap[tableID].resize(maxOffset);
        }
    }

    void addNeighbor(table_id_t tableID, offset_t nodeOffset, nodeID_t neighbor) {
        adjacencyMap[tableID][nodeOffset].push_back(neighbor);
    }

    void finalize() {
        for (auto& [tableID, lists] : adjacencyMap) {
            for (auto& neighbors : lists) {
                std::sort(neighbors.begin(), neighbors.end(),
                    [](const nodeID_t& a, const nodeID_t& b) {
                        if (a.tableID != b.tableID) {
                            return a.tableID < b.tableID;
                        }
                        return a.offset < b.offset;
                    });
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
class RadialBuildAdjacencyEdgeCompute : public EdgeCompute {
public:
    explicit RadialBuildAdjacencyEdgeCompute(RadialAdjacencyLists* adjLists) : adjLists{adjLists} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, NbrScanState::Chunk& chunk,
        bool) override {
        chunk.forEach([&](auto nbrNodes, auto, auto i) {
            adjLists->addNeighbor(boundNodeID.tableID, boundNodeID.offset, nbrNodes[i]);
        });
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<RadialBuildAdjacencyEdgeCompute>(adjLists);
    }

private:
    RadialAdjacencyLists* adjLists;
};

class RadialBuildAdjacencyAuxiliaryState : public GDSAuxiliaryState {
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

    // Get parameters from bind data
    auto radialBindData = input.bindData->constPtrCast<RadialLayoutBindData>();
    auto& config = radialBindData->optionalParams->constCast<RadialOptionalParams>();
    double minRadius = config.minRadius.getParamVal();
    double radiusIncrement = config.radiusIncrement.getParamVal();

    // Initialize position values
    auto xValues = RadialLayoutXValues(maxOffsetMap, mm);
    auto yValues = RadialLayoutYValues(maxOffsetMap, mm);

    // Count nodes and find highest-degree node as center
    uint64_t numNodes = 0;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        numNodes += maxOffset;
    }

    if (numNodes == 0) {
        auto outputVC =
            std::make_unique<RadialLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
        sharedState->factorizedTablePool.mergeLocalTables();
        return 0;
    }

    // Build adjacency lists
    auto adjLists = RadialAdjacencyLists(maxOffsetMap);
    {
        auto currentFrontier = DenseFrontier::getUnvisitedFrontier(input.context, graph);
        auto nextFrontier = DenseFrontier::getVisitedFrontier(input.context, graph,
            sharedState->getGraphNodeMaskMap());
        auto frontierPair =
            std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
        frontierPair->setActiveNodesForNextIter();
        auto edgeCompute = std::make_unique<RadialBuildAdjacencyEdgeCompute>(&adjLists);
        auto auxiliaryState = std::make_unique<RadialBuildAdjacencyAuxiliaryState>();
        auto computeState =
            GDSComputeState(std::move(frontierPair), std::move(edgeCompute), std::move(auxiliaryState));
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1 /* maxIters */);
    }
    adjLists.finalize();

    // Build flat node list and find highest-degree node
    struct NodeRef {
        table_id_t tableID;
        offset_t offset;
    };
    std::vector<NodeRef> nodeList;
    nodeList.reserve(numNodes);
    NodeRef centerNode = {0, 0};
    size_t maxDegree = 0;

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        for (offset_t i = 0; i < maxOffset; ++i) {
            nodeList.push_back({tableID, i});
            size_t degree = adjLists.getNeighbors(tableID, i).size();
            if (degree > maxDegree) {
                maxDegree = degree;
                centerNode = {tableID, i};
            }
        }
    }

    // BFS to compute distances from center
    std::vector<int64_t> distances(numNodes, -1);
    auto getNodeIndex = [&](table_id_t tableID, offset_t offset) -> size_t {
        for (size_t i = 0; i < nodeList.size(); ++i) {
            if (nodeList[i].tableID == tableID && nodeList[i].offset == offset) {
                return i;
            }
        }
        return numNodes; // Not found
    };

    size_t centerIdx = getNodeIndex(centerNode.tableID, centerNode.offset);
    distances[centerIdx] = 0;

    std::queue<size_t> bfsQueue;
    bfsQueue.push(centerIdx);

    int64_t maxDistance = 0;

    while (!bfsQueue.empty()) {
        size_t currentIdx = bfsQueue.front();
        bfsQueue.pop();

        const auto& neighbors =
            adjLists.getNeighbors(nodeList[currentIdx].tableID, nodeList[currentIdx].offset);
        for (const auto& nbr : neighbors) {
            size_t nbrIdx = getNodeIndex(nbr.tableID, nbr.offset);
            if (nbrIdx < numNodes && distances[nbrIdx] == -1) {
                distances[nbrIdx] = distances[currentIdx] + 1;
                maxDistance = std::max(maxDistance, distances[nbrIdx]);
                bfsQueue.push(nbrIdx);
            }
        }
    }

    // Group nodes by distance (ring)
    std::vector<std::vector<size_t>> rings(maxDistance + 1);
    for (size_t i = 0; i < numNodes; ++i) {
        if (distances[i] >= 0) {
            rings[distances[i]].push_back(i);
        } else {
            // Disconnected node - place in outermost ring
            rings[maxDistance].push_back(i);
        }
    }

    // Place center node at origin
    xValues.pinTable(centerNode.tableID);
    yValues.pinTable(centerNode.tableID);
    xValues.setValue(centerNode.offset, 0.0);
    yValues.setValue(centerNode.offset, 0.0);

    // Place nodes in concentric circles
    for (int64_t ring = 1; ring <= maxDistance; ++ring) {
        if (rings[ring].empty()) {
            continue;
        }

        double radius = minRadius + (ring - 1) * radiusIncrement;
        size_t nodesInRing = rings[ring].size();
        double angleStep = 2.0 * M_PI / static_cast<double>(nodesInRing);

        for (size_t i = 0; i < nodesInRing; ++i) {
            size_t nodeIdx = rings[ring][i];
            double angle = static_cast<double>(i) * angleStep;
            double x = radius * std::cos(angle);
            double y = radius * std::sin(angle);

            xValues.pinTable(nodeList[nodeIdx].tableID);
            yValues.pinTable(nodeList[nodeIdx].tableID);
            xValues.setValue(nodeList[nodeIdx].offset, x);
            yValues.setValue(nodeList[nodeIdx].offset, y);
        }
    }

    // Output results
    auto outputVC =
        std::make_unique<RadialLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
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
    columns.push_back(input->binder->createVariable(X_COLUMN_NAME, LogicalType::DOUBLE()));
    columns.push_back(input->binder->createVariable(Y_COLUMN_NAME, LogicalType::DOUBLE()));

    return std::make_unique<RadialLayoutBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput, std::make_unique<RadialOptionalParams>(input->optionalParamsLegacy));
}

function_set RadialLayoutFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(RadialLayoutFunction::name,
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
