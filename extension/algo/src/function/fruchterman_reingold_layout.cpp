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

// Default parameter values
static constexpr int64_t DEFAULT_ITERATIONS = 500;
static constexpr double DEFAULT_AREA = 10000.0;
static constexpr double DEFAULT_GRAVITY = 1.0;
static constexpr double DEFAULT_SPEED = 1.0;

// Bind data for Fruchterman-Reingold layout
struct FruchtermanReingoldBindData final : public GDSBindData {
    FruchtermanReingoldBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<FruchtermanReingoldBindData>(*this);
    }
};

// Stores per-node X coordinate values
class FRLayoutXValues {
public:
    FRLayoutXValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class FRLayoutYValues {
public:
    FRLayoutYValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class FRLayoutResultVertexCompute : public GDSResultVertexCompute {
public:
    FRLayoutResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        FRLayoutXValues& xValues, FRLayoutYValues& yValues)
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
        return std::make_unique<FRLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
    }

private:
    FRLayoutXValues& xValues;
    FRLayoutYValues& yValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> xVector;
    std::unique_ptr<ValueVector> yVector;
};

// Simple adjacency list for the FR algorithm
class FRAdjacencyLists {
public:
    FRAdjacencyLists(table_id_map_t<offset_t> maxOffsetMap) {
        for (const auto& [tableID, maxOffset] : maxOffsetMap) {
            adjacencyMap[tableID].resize(maxOffset);
        }
    }

    void addNeighbor(table_id_t tableID, offset_t nodeOffset, nodeID_t neighbor) {
        adjacencyMap[tableID][nodeOffset].push_back(neighbor);
    }

    void finalize() {
        // Sort and deduplicate
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
class FRBuildAdjacencyEdgeCompute : public EdgeCompute {
public:
    explicit FRBuildAdjacencyEdgeCompute(FRAdjacencyLists* adjLists) : adjLists{adjLists} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, NbrScanState::Chunk& chunk,
        bool) override {
        chunk.forEach([&](auto nbrNodes, auto, auto i) {
            adjLists->addNeighbor(boundNodeID.tableID, boundNodeID.offset, nbrNodes[i]);
        });
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<FRBuildAdjacencyEdgeCompute>(adjLists);
    }

private:
    FRAdjacencyLists* adjLists;
};

class FRBuildAdjacencyAuxiliaryState : public GDSAuxiliaryState {
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

    // Parameters (using defaults)
    int64_t iterations = DEFAULT_ITERATIONS;
    double area = DEFAULT_AREA;
    double gravity = DEFAULT_GRAVITY;
    double speed = DEFAULT_SPEED;

    // Count total nodes
    uint64_t numNodes = 0;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        numNodes += maxOffset;
    }

    if (numNodes == 0) {
        auto xValues = FRLayoutXValues(maxOffsetMap, mm);
        auto yValues = FRLayoutYValues(maxOffsetMap, mm);
        auto outputVC =
            std::make_unique<FRLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
        sharedState->factorizedTablePool.mergeLocalTables();
        return 0;
    }

    // Build adjacency lists
    auto adjLists = FRAdjacencyLists(maxOffsetMap);
    {
        auto currentFrontier = DenseFrontier::getUnvisitedFrontier(input.context, graph);
        auto nextFrontier = DenseFrontier::getVisitedFrontier(input.context, graph,
            sharedState->getGraphNodeMaskMap());
        auto frontierPair =
            std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
        frontierPair->setActiveNodesForNextIter();
        auto edgeCompute = std::make_unique<FRBuildAdjacencyEdgeCompute>(&adjLists);
        auto auxiliaryState = std::make_unique<FRBuildAdjacencyAuxiliaryState>();
        auto computeState =
            GDSComputeState(std::move(frontierPair), std::move(edgeCompute), std::move(auxiliaryState));
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1 /* maxIters */);
    }
    adjLists.finalize();

    // Initialize positions randomly
    auto xValues = FRLayoutXValues(maxOffsetMap, mm);
    auto yValues = FRLayoutYValues(maxOffsetMap, mm);

    double width = std::sqrt(area);
    double height = std::sqrt(area);
    std::mt19937_64 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        xValues.pinTable(tableID);
        yValues.pinTable(tableID);
        for (offset_t i = 0; i < maxOffset; ++i) {
            xValues.setValue(i, dist(rng) * width);
            yValues.setValue(i, dist(rng) * height);
        }
    }

    // Fruchterman-Reingold algorithm
    double k = std::sqrt(area / numNodes); // Optimal distance
    double k2 = k * k;
    double temperature = width / 10.0;
    double cooling = temperature / static_cast<double>(iterations);

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

    // Displacement vectors
    std::vector<double> dispX(numNodes, 0.0);
    std::vector<double> dispY(numNodes, 0.0);

    // Center of layout
    double centerX = width / 2.0;
    double centerY = height / 2.0;

    for (int64_t iter = 0; iter < iterations; ++iter) {
        // Reset displacements
        std::fill(dispX.begin(), dispX.end(), 0.0);
        std::fill(dispY.begin(), dispY.end(), 0.0);

        // Calculate repulsive forces between all pairs
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
                double dist2 = dx * dx + dy * dy;
                double distance = std::sqrt(dist2);

                if (distance > 0.0001) {
                    // Repulsive force: k^2 / distance
                    double force = k2 / distance;
                    double fx = (dx / distance) * force;
                    double fy = (dy / distance) * force;
                    dispX[i] += fx;
                    dispY[i] += fy;
                    dispX[j] -= fx;
                    dispY[j] -= fy;
                }
            }
        }

        // Calculate attractive forces along edges
        for (size_t i = 0; i < nodeList.size(); ++i) {
            xValues.pinTable(nodeList[i].tableID);
            yValues.pinTable(nodeList[i].tableID);
            double xi = xValues.getValue(nodeList[i].offset);
            double yi = yValues.getValue(nodeList[i].offset);

            const auto& neighbors = adjLists.getNeighbors(nodeList[i].tableID, nodeList[i].offset);
            for (const auto& nbr : neighbors) {
                // Find neighbor index
                size_t j = 0;
                for (; j < nodeList.size(); ++j) {
                    if (nodeList[j].tableID == nbr.tableID && nodeList[j].offset == nbr.offset) {
                        break;
                    }
                }
                if (j >= nodeList.size() || j <= i) {
                    continue; // Skip if not found or already processed
                }

                xValues.pinTable(nbr.tableID);
                yValues.pinTable(nbr.tableID);
                double xj = xValues.getValue(nbr.offset);
                double yj = yValues.getValue(nbr.offset);

                double dx = xi - xj;
                double dy = yi - yj;
                double distance = std::sqrt(dx * dx + dy * dy);

                if (distance > 0.0001) {
                    // Attractive force: distance^2 / k
                    double force = distance * distance / k;
                    double fx = (dx / distance) * force;
                    double fy = (dy / distance) * force;
                    dispX[i] -= fx;
                    dispY[i] -= fy;
                    dispX[j] += fx;
                    dispY[j] += fy;
                }
            }
        }

        // Apply gravity toward center
        for (size_t i = 0; i < nodeList.size(); ++i) {
            xValues.pinTable(nodeList[i].tableID);
            yValues.pinTable(nodeList[i].tableID);
            double xi = xValues.getValue(nodeList[i].offset);
            double yi = yValues.getValue(nodeList[i].offset);

            double dx = xi - centerX;
            double dy = yi - centerY;
            double distance = std::sqrt(dx * dx + dy * dy);

            if (distance > 0.0001) {
                double gravityForce = gravity * distance * 0.01;
                dispX[i] -= (dx / distance) * gravityForce;
                dispY[i] -= (dy / distance) * gravityForce;
            }
        }

        // Apply displacements with temperature limiting
        for (size_t i = 0; i < nodeList.size(); ++i) {
            double dispLen = std::sqrt(dispX[i] * dispX[i] + dispY[i] * dispY[i]);
            if (dispLen > 0.0001) {
                double limitedDisp = std::min(dispLen, temperature) * speed;
                double newX = xValues.getValue(nodeList[i].offset) + (dispX[i] / dispLen) * limitedDisp;
                double newY = yValues.getValue(nodeList[i].offset) + (dispY[i] / dispLen) * limitedDisp;

                // Clamp to bounds
                newX = std::max(0.0, std::min(width, newX));
                newY = std::max(0.0, std::min(height, newY));

                xValues.pinTable(nodeList[i].tableID);
                yValues.pinTable(nodeList[i].tableID);
                xValues.setValue(nodeList[i].offset, newX);
                yValues.setValue(nodeList[i].offset, newY);
            }
        }

        // Cool down
        temperature = std::max(0.0, temperature - cooling);
    }

    // Output results
    auto outputVC =
        std::make_unique<FRLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
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

    return std::make_unique<FruchtermanReingoldBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput);
}

function_set FruchtermanReingoldFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(FruchtermanReingoldFunction::name,
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
