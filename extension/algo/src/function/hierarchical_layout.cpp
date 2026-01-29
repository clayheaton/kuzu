#include "binder/binder.h"
#include "common/string_utils.h"
#include "function/algo_function.h"
#include "function/config/hierarchical_config.h"
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
#include <map>
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

// Optional parameters for Hierarchical layout
struct HierarchicalOptionalParams final : public function::OptionalParams {
    function::OptionalParam<HierarchicalRankSep> rankSep;
    function::OptionalParam<HierarchicalNodeSep> nodeSep;
    function::OptionalParam<HierarchicalIterations> iterations;

    explicit HierarchicalOptionalParams(const binder::expression_vector& optionalParams) {
        for (auto& optionalParam : optionalParams) {
            auto paramName = StringUtils::getLower(optionalParam->getAlias());
            if (paramName == HierarchicalRankSep::NAME) {
                rankSep = function::OptionalParam<HierarchicalRankSep>(optionalParam);
            } else if (paramName == HierarchicalNodeSep::NAME) {
                nodeSep = function::OptionalParam<HierarchicalNodeSep>(optionalParam);
            } else if (paramName == HierarchicalIterations::NAME) {
                iterations = function::OptionalParam<HierarchicalIterations>(optionalParam);
            } else {
                throw BinderException{"Unknown optional parameter: " + optionalParam->getAlias()};
            }
        }
    }

    // For copy only
    HierarchicalOptionalParams(function::OptionalParam<HierarchicalRankSep> rankSep,
        function::OptionalParam<HierarchicalNodeSep> nodeSep,
        function::OptionalParam<HierarchicalIterations> iterations)
        : rankSep{std::move(rankSep)}, nodeSep{std::move(nodeSep)},
          iterations{std::move(iterations)} {}

    void evaluateParams(main::ClientContext* context) override {
        rankSep.evaluateParam(context);
        nodeSep.evaluateParam(context);
        iterations.evaluateParam(context);
    }

    std::unique_ptr<function::OptionalParams> copy() override {
        return std::make_unique<HierarchicalOptionalParams>(rankSep, nodeSep, iterations);
    }
};

// Bind data for Hierarchical layout
struct HierarchicalLayoutBindData final : public GDSBindData {
    HierarchicalLayoutBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<HierarchicalOptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {
        this->optionalParams = std::move(optionalParams);
    }

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<HierarchicalLayoutBindData>(*this);
    }
};

// Stores per-node X coordinate values
class HierarchicalLayoutXValues {
public:
    HierarchicalLayoutXValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class HierarchicalLayoutYValues {
public:
    HierarchicalLayoutYValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class HierarchicalLayoutResultVertexCompute : public GDSResultVertexCompute {
public:
    HierarchicalLayoutResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        HierarchicalLayoutXValues& xValues, HierarchicalLayoutYValues& yValues)
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
        return std::make_unique<HierarchicalLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
    }

private:
    HierarchicalLayoutXValues& xValues;
    HierarchicalLayoutYValues& yValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> xVector;
    std::unique_ptr<ValueVector> yVector;
};

// Directed adjacency lists for hierarchical algorithm
class HierarchicalAdjacencyLists {
public:
    HierarchicalAdjacencyLists(table_id_map_t<offset_t> maxOffsetMap) {
        for (const auto& [tableID, maxOffset] : maxOffsetMap) {
            outgoingMap[tableID].resize(maxOffset);
            incomingMap[tableID].resize(maxOffset);
        }
    }

    void addEdge(table_id_t srcTableID, offset_t srcOffset, nodeID_t dst) {
        outgoingMap[srcTableID][srcOffset].push_back(dst);
        incomingMap[dst.tableID][dst.offset].push_back({srcOffset, srcTableID});
    }

    void finalize() {
        // Sort and deduplicate outgoing
        for (auto& [tableID, lists] : outgoingMap) {
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
        // Sort and deduplicate incoming
        for (auto& [tableID, lists] : incomingMap) {
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

    const std::vector<nodeID_t>& getOutgoing(table_id_t tableID, offset_t offset) const {
        return outgoingMap.at(tableID)[offset];
    }

    const std::vector<nodeID_t>& getIncoming(table_id_t tableID, offset_t offset) const {
        return incomingMap.at(tableID)[offset];
    }

    bool hasIncoming(table_id_t tableID, offset_t offset) const {
        return !incomingMap.at(tableID)[offset].empty();
    }

private:
    table_id_map_t<std::vector<std::vector<nodeID_t>>> outgoingMap;
    table_id_map_t<std::vector<std::vector<nodeID_t>>> incomingMap;
};

// Edge compute to build directed adjacency lists
class HierarchicalBuildAdjacencyEdgeCompute : public EdgeCompute {
public:
    explicit HierarchicalBuildAdjacencyEdgeCompute(HierarchicalAdjacencyLists* adjLists)
        : adjLists{adjLists} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, NbrScanState::Chunk& chunk,
        bool) override {
        chunk.forEach([&](auto nbrNodes, auto, auto i) {
            adjLists->addEdge(boundNodeID.tableID, boundNodeID.offset, nbrNodes[i]);
        });
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<HierarchicalBuildAdjacencyEdgeCompute>(adjLists);
    }

private:
    HierarchicalAdjacencyLists* adjLists;
};

class HierarchicalBuildAdjacencyAuxiliaryState : public GDSAuxiliaryState {
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
    auto hBindData = input.bindData->constPtrCast<HierarchicalLayoutBindData>();
    auto& config = hBindData->optionalParams->constCast<HierarchicalOptionalParams>();
    double rankSep = config.rankSep.getParamVal();
    double nodeSep = config.nodeSep.getParamVal();
    int64_t iterations = config.iterations.getParamVal();

    // Count total nodes
    uint64_t numNodes = 0;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        numNodes += maxOffset;
    }

    // Initialize coordinate storage
    auto xValues = HierarchicalLayoutXValues(maxOffsetMap, mm);
    auto yValues = HierarchicalLayoutYValues(maxOffsetMap, mm);

    if (numNodes == 0) {
        auto outputVC =
            std::make_unique<HierarchicalLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
        sharedState->factorizedTablePool.mergeLocalTables();
        return 0;
    }

    // Build directed adjacency lists using FWD direction
    auto adjLists = HierarchicalAdjacencyLists(maxOffsetMap);
    {
        auto currentFrontier = DenseFrontier::getUnvisitedFrontier(input.context, graph);
        auto nextFrontier = DenseFrontier::getVisitedFrontier(input.context, graph,
            sharedState->getGraphNodeMaskMap());
        auto frontierPair =
            std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
        frontierPair->setActiveNodesForNextIter();
        auto edgeCompute = std::make_unique<HierarchicalBuildAdjacencyEdgeCompute>(&adjLists);
        auto auxiliaryState = std::make_unique<HierarchicalBuildAdjacencyAuxiliaryState>();
        auto computeState =
            GDSComputeState(std::move(frontierPair), std::move(edgeCompute), std::move(auxiliaryState));
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::FWD,
            1 /* maxIters */);
    }
    adjLists.finalize();

    // Build flat node list for iteration
    struct NodeRef {
        table_id_t tableID;
        offset_t offset;
    };
    std::vector<NodeRef> nodeList;
    nodeList.reserve(numNodes);
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        for (offset_t i = 0; i < maxOffset; ++i) {
            nodeList.push_back({tableID, i});
        }
    }

    // Create node index map for quick lookup
    std::map<std::pair<table_id_t, offset_t>, size_t> nodeIndexMap;
    for (size_t i = 0; i < nodeList.size(); ++i) {
        nodeIndexMap[{nodeList[i].tableID, nodeList[i].offset}] = i;
    }

    // Step 1: Assign nodes to layers using longest path from sources
    std::vector<int64_t> layer(numNodes, -1);
    int64_t maxLayer = 0;

    // Find sources (nodes with no incoming edges)
    std::queue<size_t> queue;
    for (size_t i = 0; i < nodeList.size(); ++i) {
        if (!adjLists.hasIncoming(nodeList[i].tableID, nodeList[i].offset)) {
            layer[i] = 0;
            queue.push(i);
        }
    }

    // If no sources found (cycle), pick first node
    if (queue.empty() && !nodeList.empty()) {
        layer[0] = 0;
        queue.push(0);
    }

    // BFS to assign layers (longest path)
    while (!queue.empty()) {
        size_t nodeIdx = queue.front();
        queue.pop();

        const auto& outgoing = adjLists.getOutgoing(nodeList[nodeIdx].tableID, nodeList[nodeIdx].offset);
        for (const auto& nbr : outgoing) {
            auto it = nodeIndexMap.find({nbr.tableID, nbr.offset});
            if (it != nodeIndexMap.end()) {
                size_t nbrIdx = it->second;
                int64_t newLayer = layer[nodeIdx] + 1;
                if (layer[nbrIdx] < newLayer) {
                    layer[nbrIdx] = newLayer;
                    maxLayer = std::max(maxLayer, newLayer);
                    queue.push(nbrIdx);
                }
            }
        }
    }

    // Assign unvisited nodes to layer 0
    for (size_t i = 0; i < numNodes; ++i) {
        if (layer[i] < 0) {
            layer[i] = 0;
        }
    }

    // Step 2: Group nodes by layer
    std::vector<std::vector<size_t>> layers(maxLayer + 1);
    for (size_t i = 0; i < numNodes; ++i) {
        layers[layer[i]].push_back(i);
    }

    // Step 3: Order nodes within layers to minimize crossings (barycenter heuristic)
    std::vector<double> position(numNodes);
    for (size_t l = 0; l <= static_cast<size_t>(maxLayer); ++l) {
        for (size_t i = 0; i < layers[l].size(); ++i) {
            position[layers[l][i]] = static_cast<double>(i);
        }
    }

    // Iterate to minimize crossings
    for (int64_t iter = 0; iter < iterations; ++iter) {
        // Forward sweep
        for (size_t l = 1; l <= static_cast<size_t>(maxLayer); ++l) {
            for (size_t nodeIdx : layers[l]) {
                const auto& incoming = adjLists.getIncoming(nodeList[nodeIdx].tableID,
                    nodeList[nodeIdx].offset);
                if (!incoming.empty()) {
                    double sum = 0.0;
                    int count = 0;
                    for (const auto& nbr : incoming) {
                        auto it = nodeIndexMap.find({nbr.tableID, nbr.offset});
                        if (it != nodeIndexMap.end()) {
                            sum += position[it->second];
                            count++;
                        }
                    }
                    if (count > 0) {
                        position[nodeIdx] = sum / count;
                    }
                }
            }
            std::sort(layers[l].begin(), layers[l].end(),
                [&position](size_t a, size_t b) { return position[a] < position[b]; });
            for (size_t i = 0; i < layers[l].size(); ++i) {
                position[layers[l][i]] = static_cast<double>(i);
            }
        }

        // Backward sweep
        for (int64_t l = maxLayer - 1; l >= 0; --l) {
            for (size_t nodeIdx : layers[l]) {
                const auto& outgoing = adjLists.getOutgoing(nodeList[nodeIdx].tableID,
                    nodeList[nodeIdx].offset);
                if (!outgoing.empty()) {
                    double sum = 0.0;
                    int count = 0;
                    for (const auto& nbr : outgoing) {
                        auto it = nodeIndexMap.find({nbr.tableID, nbr.offset});
                        if (it != nodeIndexMap.end()) {
                            sum += position[it->second];
                            count++;
                        }
                    }
                    if (count > 0) {
                        position[nodeIdx] = sum / count;
                    }
                }
            }
            std::sort(layers[l].begin(), layers[l].end(),
                [&position](size_t a, size_t b) { return position[a] < position[b]; });
            for (size_t i = 0; i < layers[l].size(); ++i) {
                position[layers[l][i]] = static_cast<double>(i);
            }
        }
    }

    // Step 4: Assign final coordinates
    for (size_t l = 0; l <= static_cast<size_t>(maxLayer); ++l) {
        for (size_t i = 0; i < layers[l].size(); ++i) {
            size_t nodeIdx = layers[l][i];
            double x = static_cast<double>(l) * rankSep;
            double y = static_cast<double>(i) * nodeSep;

            xValues.pinTable(nodeList[nodeIdx].tableID);
            yValues.pinTable(nodeList[nodeIdx].tableID);
            xValues.setValue(nodeList[nodeIdx].offset, x);
            yValues.setValue(nodeList[nodeIdx].offset, y);
        }
    }

    // Output results
    auto outputVC =
        std::make_unique<HierarchicalLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
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

    return std::make_unique<HierarchicalLayoutBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput, std::make_unique<HierarchicalOptionalParams>(input->optionalParamsLegacy));
}

function_set HierarchicalLayoutFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(HierarchicalLayoutFunction::name,
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
