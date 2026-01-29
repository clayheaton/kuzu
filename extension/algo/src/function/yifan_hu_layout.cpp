#include "binder/binder.h"
#include "common/string_utils.h"
#include "function/algo_function.h"
#include "function/config/yifan_hu_config.h"
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
#include <limits>
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

// Optional parameters for Yifan Hu layout
struct YHOptionalParams final : public function::OptionalParams {
    function::OptionalParam<YHIterations> iterations;
    function::OptionalParam<YHOptimalDistance> K;
    function::OptionalParam<YHRelativeStrength> C;
    function::OptionalParam<YHStepSize> stepSize;
    function::OptionalParam<YHStepRatio> stepRatio;
    function::OptionalParam<YHConvergenceThreshold> convergenceThreshold;

    explicit YHOptionalParams(const binder::expression_vector& optionalParams) {
        for (auto& optionalParam : optionalParams) {
            auto paramName = StringUtils::getLower(optionalParam->getAlias());
            if (paramName == YHIterations::NAME) {
                iterations = function::OptionalParam<YHIterations>(optionalParam);
            } else if (paramName == YHOptimalDistance::NAME) {
                K = function::OptionalParam<YHOptimalDistance>(optionalParam);
            } else if (paramName == YHRelativeStrength::NAME) {
                C = function::OptionalParam<YHRelativeStrength>(optionalParam);
            } else if (paramName == YHStepSize::NAME) {
                stepSize = function::OptionalParam<YHStepSize>(optionalParam);
            } else if (paramName == YHStepRatio::NAME) {
                stepRatio = function::OptionalParam<YHStepRatio>(optionalParam);
            } else if (paramName == YHConvergenceThreshold::NAME) {
                convergenceThreshold = function::OptionalParam<YHConvergenceThreshold>(optionalParam);
            } else {
                throw BinderException{"Unknown optional parameter: " + optionalParam->getAlias()};
            }
        }
    }

    // For copy only
    YHOptionalParams(function::OptionalParam<YHIterations> iterations,
        function::OptionalParam<YHOptimalDistance> K,
        function::OptionalParam<YHRelativeStrength> C,
        function::OptionalParam<YHStepSize> stepSize,
        function::OptionalParam<YHStepRatio> stepRatio,
        function::OptionalParam<YHConvergenceThreshold> convergenceThreshold)
        : iterations{std::move(iterations)}, K{std::move(K)}, C{std::move(C)},
          stepSize{std::move(stepSize)}, stepRatio{std::move(stepRatio)},
          convergenceThreshold{std::move(convergenceThreshold)} {}

    void evaluateParams(main::ClientContext* context) override {
        iterations.evaluateParam(context);
        K.evaluateParam(context);
        C.evaluateParam(context);
        stepSize.evaluateParam(context);
        stepRatio.evaluateParam(context);
        convergenceThreshold.evaluateParam(context);
    }

    std::unique_ptr<function::OptionalParams> copy() override {
        return std::make_unique<YHOptionalParams>(iterations, K, C, stepSize, stepRatio,
            convergenceThreshold);
    }
};

// Bind data for Yifan Hu layout
struct YifanHuBindData final : public GDSBindData {
    YifanHuBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<YHOptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {
        this->optionalParams = std::move(optionalParams);
    }

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<YifanHuBindData>(*this);
    }
};

// Stores per-node X coordinate values
class YHLayoutXValues {
public:
    YHLayoutXValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class YHLayoutYValues {
public:
    YHLayoutYValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class YHLayoutResultVertexCompute : public GDSResultVertexCompute {
public:
    YHLayoutResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        YHLayoutXValues& xValues, YHLayoutYValues& yValues)
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
        return std::make_unique<YHLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
    }

private:
    YHLayoutXValues& xValues;
    YHLayoutYValues& yValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> xVector;
    std::unique_ptr<ValueVector> yVector;
};

// Simple adjacency list for the Yifan Hu algorithm
class YHAdjacencyLists {
public:
    YHAdjacencyLists(table_id_map_t<offset_t> maxOffsetMap) {
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
class YHBuildAdjacencyEdgeCompute : public EdgeCompute {
public:
    explicit YHBuildAdjacencyEdgeCompute(YHAdjacencyLists* adjLists) : adjLists{adjLists} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, NbrScanState::Chunk& chunk,
        bool) override {
        chunk.forEach([&](auto nbrNodes, auto, auto i) {
            adjLists->addNeighbor(boundNodeID.tableID, boundNodeID.offset, nbrNodes[i]);
        });
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<YHBuildAdjacencyEdgeCompute>(adjLists);
    }

private:
    YHAdjacencyLists* adjLists;
};

class YHBuildAdjacencyAuxiliaryState : public GDSAuxiliaryState {
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
    auto yhBindData = input.bindData->constPtrCast<YifanHuBindData>();
    auto& config = yhBindData->optionalParams->constCast<YHOptionalParams>();
    int64_t iterations = config.iterations.getParamVal();
    double K = config.K.getParamVal();
    double C = config.C.getParamVal();
    double stepSize = config.stepSize.getParamVal();
    double stepRatio = config.stepRatio.getParamVal();
    double convergenceThreshold = config.convergenceThreshold.getParamVal();

    // Count total nodes
    uint64_t numNodes = 0;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        numNodes += maxOffset;
    }

    if (numNodes == 0) {
        auto xValues = YHLayoutXValues(maxOffsetMap, mm);
        auto yValues = YHLayoutYValues(maxOffsetMap, mm);
        auto outputVC =
            std::make_unique<YHLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
        sharedState->factorizedTablePool.mergeLocalTables();
        return 0;
    }

    // Build adjacency lists
    auto adjLists = YHAdjacencyLists(maxOffsetMap);
    {
        auto currentFrontier = DenseFrontier::getUnvisitedFrontier(input.context, graph);
        auto nextFrontier = DenseFrontier::getVisitedFrontier(input.context, graph,
            sharedState->getGraphNodeMaskMap());
        auto frontierPair =
            std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
        frontierPair->setActiveNodesForNextIter();
        auto edgeCompute = std::make_unique<YHBuildAdjacencyEdgeCompute>(&adjLists);
        auto auxiliaryState = std::make_unique<YHBuildAdjacencyAuxiliaryState>();
        auto computeState =
            GDSComputeState(std::move(frontierPair), std::move(edgeCompute), std::move(auxiliaryState));
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1 /* maxIters */);
    }
    adjLists.finalize();

    // Initialize positions randomly
    auto xValues = YHLayoutXValues(maxOffsetMap, mm);
    auto yValues = YHLayoutYValues(maxOffsetMap, mm);

    double K2 = K * K;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-K * std::sqrt(static_cast<double>(numNodes)) / 2.0,
                                                  K * std::sqrt(static_cast<double>(numNodes)) / 2.0);

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

    // Displacement vectors
    std::vector<double> dispX(numNodes, 0.0);
    std::vector<double> dispY(numNodes, 0.0);

    double prevEnergy = std::numeric_limits<double>::max();

    // Yifan Hu algorithm
    for (int64_t iter = 0; iter < iterations; ++iter) {
        std::fill(dispX.begin(), dispX.end(), 0.0);
        std::fill(dispY.begin(), dispY.end(), 0.0);

        double energy = 0.0;

        // Repulsive forces: F_rep = C * K^2 / distance
        for (size_t i = 0; i < nodeList.size(); ++i) {
            xValues.pinTable(nodeList[i].tableID);
            yValues.pinTable(nodeList[i].tableID);
            double xi = xValues.getValue(nodeList[i].offset);
            double yi = yValues.getValue(nodeList[i].offset);

            for (size_t j = i + 1; j < nodeList.size(); ++j) {
                xValues.pinTable(nodeList[j].tableID);
                yValues.pinTable(nodeList[j].tableID);
                double xj = xValues.getValue(nodeList[j].offset);
                double yj = yValues.getValue(nodeList[j].offset);

                double dx = xi - xj;
                double dy = yi - yj;
                double distance = std::sqrt(dx * dx + dy * dy);

                if (distance > 0.0001) {
                    double force = C * K2 / distance;
                    double fx = (dx / distance) * force;
                    double fy = (dy / distance) * force;
                    dispX[i] += fx;
                    dispY[i] += fy;
                    dispX[j] -= fx;
                    dispY[j] -= fy;
                }
            }
        }

        // Attractive forces: F_att = distance^2 / K
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
                    double force = distance * distance / K;
                    double fx = (dx / distance) * force;
                    double fy = (dy / distance) * force;
                    dispX[i] -= fx;
                    dispY[i] -= fy;
                    dispX[j] += fx;
                    dispY[j] += fy;
                }
            }
        }

        // Apply displacements with step size limiting
        for (size_t i = 0; i < nodeList.size(); ++i) {
            double dispLen = std::sqrt(dispX[i] * dispX[i] + dispY[i] * dispY[i]);
            if (dispLen > 0.0001) {
                double limitedDisp = std::min(dispLen, stepSize);
                xValues.pinTable(nodeList[i].tableID);
                yValues.pinTable(nodeList[i].tableID);
                double newX = xValues.getValue(nodeList[i].offset) + (dispX[i] / dispLen) * limitedDisp;
                double newY = yValues.getValue(nodeList[i].offset) + (dispY[i] / dispLen) * limitedDisp;
                xValues.setValue(nodeList[i].offset, newX);
                yValues.setValue(nodeList[i].offset, newY);
                energy += dispLen * dispLen;
            }
        }

        // Adaptive step size
        if (energy < prevEnergy) {
            stepSize *= stepRatio;
        } else {
            stepSize *= stepRatio * stepRatio;
        }

        // Convergence check
        double energyChange = std::abs(prevEnergy - energy) / (energy + 1e-10);
        if (energyChange < convergenceThreshold && iter > 10) {
            break;
        }

        prevEnergy = energy;
    }

    // Output results
    auto outputVC =
        std::make_unique<YHLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
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

    return std::make_unique<YifanHuBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput, std::make_unique<YHOptionalParams>(input->optionalParamsLegacy));
}

function_set YifanHuLayoutFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(YifanHuLayoutFunction::name,
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
