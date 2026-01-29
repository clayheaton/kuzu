#include "binder/binder.h"
#include "common/string_utils.h"
#include "function/algo_function.h"
#include "function/config/forceatlas2_config.h"
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
#include <random>
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

// Optional parameters for ForceAtlas2 layout
struct FA2OptionalParams final : public function::OptionalParams {
    function::OptionalParam<FA2Iterations> iterations;
    function::OptionalParam<FA2ScalingRatio> scalingRatio;
    function::OptionalParam<FA2Gravity> gravity;
    function::OptionalParam<FA2JitterTolerance> jitterTolerance;

    explicit FA2OptionalParams(const binder::expression_vector& optionalParams) {
        for (auto& optionalParam : optionalParams) {
            auto paramName = StringUtils::getLower(optionalParam->getAlias());
            if (paramName == FA2Iterations::NAME) {
                iterations = function::OptionalParam<FA2Iterations>(optionalParam);
            } else if (paramName == FA2ScalingRatio::NAME) {
                scalingRatio = function::OptionalParam<FA2ScalingRatio>(optionalParam);
            } else if (paramName == FA2Gravity::NAME) {
                gravity = function::OptionalParam<FA2Gravity>(optionalParam);
            } else if (paramName == FA2JitterTolerance::NAME) {
                jitterTolerance = function::OptionalParam<FA2JitterTolerance>(optionalParam);
            } else {
                throw BinderException{"Unknown optional parameter: " + optionalParam->getAlias()};
            }
        }
    }

    // For copy only
    FA2OptionalParams(function::OptionalParam<FA2Iterations> iterations,
        function::OptionalParam<FA2ScalingRatio> scalingRatio,
        function::OptionalParam<FA2Gravity> gravity,
        function::OptionalParam<FA2JitterTolerance> jitterTolerance)
        : iterations{std::move(iterations)}, scalingRatio{std::move(scalingRatio)},
          gravity{std::move(gravity)}, jitterTolerance{std::move(jitterTolerance)} {}

    void evaluateParams(main::ClientContext* context) override {
        iterations.evaluateParam(context);
        scalingRatio.evaluateParam(context);
        gravity.evaluateParam(context);
        jitterTolerance.evaluateParam(context);
    }

    std::unique_ptr<function::OptionalParams> copy() override {
        return std::make_unique<FA2OptionalParams>(iterations, scalingRatio, gravity, jitterTolerance);
    }
};

// Bind data for ForceAtlas2 layout
struct ForceAtlas2BindData final : public GDSBindData {
    ForceAtlas2BindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<FA2OptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {
        this->optionalParams = std::move(optionalParams);
    }

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<ForceAtlas2BindData>(*this);
    }
};

// Stores per-node X coordinate values
class FA2LayoutXValues {
public:
    FA2LayoutXValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class FA2LayoutYValues {
public:
    FA2LayoutYValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class FA2LayoutResultVertexCompute : public GDSResultVertexCompute {
public:
    FA2LayoutResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        FA2LayoutXValues& xValues, FA2LayoutYValues& yValues)
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
        return std::make_unique<FA2LayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
    }

private:
    FA2LayoutXValues& xValues;
    FA2LayoutYValues& yValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> xVector;
    std::unique_ptr<ValueVector> yVector;
};

// Adjacency list for the ForceAtlas2 algorithm (also tracks degrees)
class FA2AdjacencyLists {
public:
    FA2AdjacencyLists(table_id_map_t<offset_t> maxOffsetMap) {
        for (const auto& [tableID, maxOffset] : maxOffsetMap) {
            adjacencyMap[tableID].resize(maxOffset);
            degreeMap[tableID].resize(maxOffset, 0);
        }
    }

    void addNeighbor(table_id_t tableID, offset_t nodeOffset, nodeID_t neighbor) {
        adjacencyMap[tableID][nodeOffset].push_back(neighbor);
    }

    void finalize() {
        for (auto& [tableID, lists] : adjacencyMap) {
            for (size_t i = 0; i < lists.size(); ++i) {
                auto& neighbors = lists[i];
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
                degreeMap[tableID][i] = static_cast<uint32_t>(neighbors.size());
            }
        }
    }

    const std::vector<nodeID_t>& getNeighbors(table_id_t tableID, offset_t offset) const {
        return adjacencyMap.at(tableID)[offset];
    }

    uint32_t getDegree(table_id_t tableID, offset_t offset) const {
        return degreeMap.at(tableID)[offset];
    }

private:
    table_id_map_t<std::vector<std::vector<nodeID_t>>> adjacencyMap;
    table_id_map_t<std::vector<uint32_t>> degreeMap;
};

// Edge compute to build adjacency lists
class FA2BuildAdjacencyEdgeCompute : public EdgeCompute {
public:
    explicit FA2BuildAdjacencyEdgeCompute(FA2AdjacencyLists* adjLists) : adjLists{adjLists} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, NbrScanState::Chunk& chunk,
        bool) override {
        chunk.forEach([&](auto nbrNodes, auto, auto i) {
            adjLists->addNeighbor(boundNodeID.tableID, boundNodeID.offset, nbrNodes[i]);
        });
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<FA2BuildAdjacencyEdgeCompute>(adjLists);
    }

private:
    FA2AdjacencyLists* adjLists;
};

class FA2BuildAdjacencyAuxiliaryState : public GDSAuxiliaryState {
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
    auto fa2BindData = input.bindData->constPtrCast<ForceAtlas2BindData>();
    auto& config = fa2BindData->optionalParams->constCast<FA2OptionalParams>();
    int64_t iterations = config.iterations.getParamVal();
    double scalingRatio = config.scalingRatio.getParamVal();
    double gravity = config.gravity.getParamVal();
    double jitterTolerance = config.jitterTolerance.getParamVal();

    // Count total nodes
    uint64_t numNodes = 0;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        numNodes += maxOffset;
    }

    if (numNodes == 0) {
        auto xValues = FA2LayoutXValues(maxOffsetMap, mm);
        auto yValues = FA2LayoutYValues(maxOffsetMap, mm);
        auto outputVC =
            std::make_unique<FA2LayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
        sharedState->factorizedTablePool.mergeLocalTables();
        return 0;
    }

    // Build adjacency lists
    auto adjLists = FA2AdjacencyLists(maxOffsetMap);
    {
        auto currentFrontier = DenseFrontier::getUnvisitedFrontier(input.context, graph);
        auto nextFrontier = DenseFrontier::getVisitedFrontier(input.context, graph,
            sharedState->getGraphNodeMaskMap());
        auto frontierPair =
            std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
        frontierPair->setActiveNodesForNextIter();
        auto edgeCompute = std::make_unique<FA2BuildAdjacencyEdgeCompute>(&adjLists);
        auto auxiliaryState = std::make_unique<FA2BuildAdjacencyAuxiliaryState>();
        auto computeState =
            GDSComputeState(std::move(frontierPair), std::move(edgeCompute), std::move(auxiliaryState));
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1 /* maxIters */);
    }
    adjLists.finalize();

    // Initialize positions randomly
    auto xValues = FA2LayoutXValues(maxOffsetMap, mm);
    auto yValues = FA2LayoutYValues(maxOffsetMap, mm);

    double width = 100.0;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-width / 2.0, width / 2.0);

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        xValues.pinTable(tableID);
        yValues.pinTable(tableID);
        for (offset_t i = 0; i < maxOffset; ++i) {
            xValues.setValue(i, dist(rng));
            yValues.setValue(i, dist(rng));
        }
    }

    // Build flat node list
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

    // ForceAtlas2 state
    std::vector<double> dispX(numNodes, 0.0);
    std::vector<double> dispY(numNodes, 0.0);
    std::vector<double> prevDispX(numNodes, 0.0);
    std::vector<double> prevDispY(numNodes, 0.0);

    double globalSpeed = 1.0;

    // ForceAtlas2 main loop
    for (int64_t iter = 0; iter < iterations; ++iter) {
        prevDispX = dispX;
        prevDispY = dispY;

        std::fill(dispX.begin(), dispX.end(), 0.0);
        std::fill(dispY.begin(), dispY.end(), 0.0);

        // Repulsive forces: F_rep = scalingRatio * (degree_i + 1) * (degree_j + 1) / distance
        for (size_t i = 0; i < nodeList.size(); ++i) {
            xValues.pinTable(nodeList[i].tableID);
            yValues.pinTable(nodeList[i].tableID);
            double xi = xValues.getValue(nodeList[i].offset);
            double yi = yValues.getValue(nodeList[i].offset);
            uint32_t degreeI = adjLists.getDegree(nodeList[i].tableID, nodeList[i].offset);

            for (size_t j = i + 1; j < nodeList.size(); ++j) {
                xValues.pinTable(nodeList[j].tableID);
                yValues.pinTable(nodeList[j].tableID);
                double xj = xValues.getValue(nodeList[j].offset);
                double yj = yValues.getValue(nodeList[j].offset);
                uint32_t degreeJ = adjLists.getDegree(nodeList[j].tableID, nodeList[j].offset);

                double dx = xi - xj;
                double dy = yi - yj;
                double distance = std::sqrt(dx * dx + dy * dy);

                if (distance > 0.0001) {
                    double factor = static_cast<double>((degreeI + 1) * (degreeJ + 1));
                    double force = scalingRatio * factor / distance;
                    double fx = (dx / distance) * force;
                    double fy = (dy / distance) * force;
                    dispX[i] += fx;
                    dispY[i] += fy;
                    dispX[j] -= fx;
                    dispY[j] -= fy;
                }
            }
        }

        // Attractive forces: F_att = distance (linear)
        for (size_t i = 0; i < nodeList.size(); ++i) {
            xValues.pinTable(nodeList[i].tableID);
            yValues.pinTable(nodeList[i].tableID);
            double xi = xValues.getValue(nodeList[i].offset);
            double yi = yValues.getValue(nodeList[i].offset);

            const auto& neighbors = adjLists.getNeighbors(nodeList[i].tableID, nodeList[i].offset);
            for (const auto& nbr : neighbors) {
                size_t j = 0;
                for (; j < nodeList.size(); ++j) {
                    if (nodeList[j].tableID == nbr.tableID && nodeList[j].offset == nbr.offset) {
                        break;
                    }
                }
                if (j >= nodeList.size() || j <= i) {
                    continue;
                }

                xValues.pinTable(nbr.tableID);
                yValues.pinTable(nbr.tableID);
                double xj = xValues.getValue(nbr.offset);
                double yj = yValues.getValue(nbr.offset);

                double dx = xi - xj;
                double dy = yi - yj;
                double distance = std::sqrt(dx * dx + dy * dy);

                if (distance > 0.0001) {
                    double force = distance;
                    double fx = (dx / distance) * force;
                    double fy = (dy / distance) * force;
                    dispX[i] -= fx;
                    dispY[i] -= fy;
                    dispX[j] += fx;
                    dispY[j] += fy;
                }
            }
        }

        // Gravity: F_grav = gravity * (degree + 1)
        for (size_t i = 0; i < nodeList.size(); ++i) {
            xValues.pinTable(nodeList[i].tableID);
            yValues.pinTable(nodeList[i].tableID);
            double xi = xValues.getValue(nodeList[i].offset);
            double yi = yValues.getValue(nodeList[i].offset);
            uint32_t degree = adjLists.getDegree(nodeList[i].tableID, nodeList[i].offset);

            double distance = std::sqrt(xi * xi + yi * yi);

            if (distance > 0.0001) {
                double gravityForce = gravity * (degree + 1);
                dispX[i] -= (xi / distance) * gravityForce;
                dispY[i] -= (yi / distance) * gravityForce;
            }
        }

        // Adaptive speed
        double totalSwing = 0.0;
        double totalTraction = 0.0;

        for (size_t i = 0; i < nodeList.size(); ++i) {
            double forceLen = std::sqrt(dispX[i] * dispX[i] + dispY[i] * dispY[i]);
            double prevForceLen = std::sqrt(prevDispX[i] * prevDispX[i] + prevDispY[i] * prevDispY[i]);

            double swingX = dispX[i] - prevDispX[i];
            double swingY = dispY[i] - prevDispY[i];
            double swing = std::sqrt(swingX * swingX + swingY * swingY);

            double traction = std::abs(forceLen + prevForceLen) / 2.0;

            uint32_t degree = adjLists.getDegree(nodeList[i].tableID, nodeList[i].offset);
            totalSwing += (degree + 1) * swing;
            totalTraction += (degree + 1) * traction;
        }

        if (totalSwing > 0.0) {
            double targetSpeed = jitterTolerance * jitterTolerance * totalTraction / totalSwing;
            globalSpeed = globalSpeed + std::min(targetSpeed - globalSpeed, 0.5 * globalSpeed);
            globalSpeed = std::max(globalSpeed, 0.01);
        }

        // Apply displacements
        for (size_t i = 0; i < nodeList.size(); ++i) {
            double dispLen = std::sqrt(dispX[i] * dispX[i] + dispY[i] * dispY[i]);
            if (dispLen > 0.0001) {
                double swingX = dispX[i] - prevDispX[i];
                double swingY = dispY[i] - prevDispY[i];
                double swing = std::sqrt(swingX * swingX + swingY * swingY);
                double nodeSpeed = globalSpeed / (1.0 + globalSpeed * swing);

                double factor = std::min(nodeSpeed * dispLen, 10.0) / dispLen;
                xValues.pinTable(nodeList[i].tableID);
                yValues.pinTable(nodeList[i].tableID);
                double newX = xValues.getValue(nodeList[i].offset) + dispX[i] * factor;
                double newY = yValues.getValue(nodeList[i].offset) + dispY[i] * factor;
                xValues.setValue(nodeList[i].offset, newX);
                yValues.setValue(nodeList[i].offset, newY);
            }
        }
    }

    // Output results
    auto outputVC =
        std::make_unique<FA2LayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
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

    return std::make_unique<ForceAtlas2BindData>(std::move(columns), std::move(graphEntry),
        nodeOutput, std::make_unique<FA2OptionalParams>(input->optionalParamsLegacy));
}

function_set ForceAtlas2Function::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(ForceAtlas2Function::name,
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
