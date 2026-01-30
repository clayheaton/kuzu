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

#include <cmath>
#include <limits>
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
static constexpr double DEFAULT_RADIUS = 100.0;
static constexpr double DEFAULT_START_ANGLE = 0.0;

// M_PI may not be defined on all platforms
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Bind data for circular layout
struct CircularLayoutBindData final : public GDSBindData {
    CircularLayoutBindData(expression_vector columns, NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<CircularLayoutBindData>(*this);
    }
};

// Stores per-node X coordinate values
class CircularLayoutXValues {
public:
    CircularLayoutXValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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
class CircularLayoutYValues {
public:
    CircularLayoutYValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm) {
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

// Output vertex compute class - writes (node, x, y) results to factorized table
class CircularLayoutResultVertexCompute : public GDSResultVertexCompute {
public:
    CircularLayoutResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        CircularLayoutXValues& xValues, CircularLayoutYValues& yValues)
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
        return std::make_unique<CircularLayoutResultVertexCompute>(mm, sharedState, xValues,
            yValues);
    }

private:
    CircularLayoutXValues& xValues;
    CircularLayoutYValues& yValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> xVector;
    std::unique_ptr<ValueVector> yVector;
};

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    auto clientContext = input.context->clientContext;
    auto transaction = transaction::Transaction::Get(*clientContext);
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto graph = sharedState->graph.get();
    auto maxOffsetMap = graph->getMaxOffsetMap(transaction);
    auto mm = MemoryManager::Get(*clientContext);

    // Use defaults
    double radius = DEFAULT_RADIUS;
    double startAngle = DEFAULT_START_ANGLE;

    // Initialize position values
    auto xValues = CircularLayoutXValues(maxOffsetMap, mm);
    auto yValues = CircularLayoutYValues(maxOffsetMap, mm);

    // Count total nodes for angle calculation
    uint64_t totalNodes = 0;
    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        totalNodes += maxOffset;
    }

    if (totalNodes == 0) {
        // Empty graph, nothing to do
        auto outputVC =
            std::make_unique<CircularLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
        sharedState->factorizedTablePool.mergeLocalTables();
        return 0;
    }

    // Compute positions around circle in order of tableID/offset
    double angleStep = 2.0 * M_PI / static_cast<double>(totalNodes);
    uint64_t nodeIndex = 0;

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        xValues.pinTable(tableID);
        yValues.pinTable(tableID);

        for (offset_t i = 0; i < maxOffset; ++i) {
            double angle = startAngle + static_cast<double>(nodeIndex) * angleStep;
            double x = radius * std::cos(angle);
            double y = radius * std::sin(angle);
            xValues.setValue(i, x);
            yValues.setValue(i, y);
            nodeIndex++;
        }
    }

    // Normalize coordinates to [0, 1] space
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        xValues.pinTable(tableID);
        yValues.pinTable(tableID);
        for (offset_t i = 0; i < maxOffset; ++i) {
            double x = xValues.getValue(i);
            double y = yValues.getValue(i);
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }

    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    // Avoid division by zero for single-node or collinear layouts
    if (rangeX < 0.0001) rangeX = 1.0;
    if (rangeY < 0.0001) rangeY = 1.0;

    for (const auto& [tableID, maxOffset] : maxOffsetMap) {
        xValues.pinTable(tableID);
        yValues.pinTable(tableID);
        for (offset_t i = 0; i < maxOffset; ++i) {
            double x = xValues.getValue(i);
            double y = yValues.getValue(i);
            xValues.setValue(i, (x - minX) / rangeX);
            yValues.setValue(i, (y - minY) / rangeY);
        }
    }

    // Output results
    auto outputVC =
        std::make_unique<CircularLayoutResultVertexCompute>(mm, sharedState, xValues, yValues);
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

    return std::make_unique<CircularLayoutBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput);
}

function_set CircularLayoutFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(CircularLayoutFunction::name,
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
