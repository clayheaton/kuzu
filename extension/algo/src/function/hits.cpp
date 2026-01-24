#include <cmath>

#include "binder/binder.h"
#include "common/exception/binder.h"
#include "common/string_utils.h"
#include "common/task_system/progress_bar.h"
#include "function/algo_function.h"
#include "function/config/max_iterations_config.h"
#include "function/config/page_rank_config.h"
#include "function/gds/gds_utils.h"
#include "function/gds/gds_vertex_compute.h"
#include "function/table/bind_input.h"
#include "processor/execution_context.h"
#include "transaction/transaction.h"

using namespace kuzu::processor;
using namespace kuzu::common;
using namespace kuzu::binder;
using namespace kuzu::storage;
using namespace kuzu::graph;
using namespace kuzu::function;

namespace kuzu {
namespace algo_extension {

struct HitsOptionalParams final : public MaxIterationOptionalParams {
    OptionalParam<Tolerance> tolerance;

    explicit HitsOptionalParams(const expression_vector& optionalParams);

    HitsOptionalParams(OptionalParam<MaxIterations> maxIterations,
        OptionalParam<Tolerance> tolerance)
        : MaxIterationOptionalParams{maxIterations}, tolerance{std::move(tolerance)} {}

    void evaluateParams(main::ClientContext* context) override {
        MaxIterationOptionalParams::evaluateParams(context);
        tolerance.evaluateParam(context);
    }

    std::unique_ptr<function::OptionalParams> copy() override {
        return std::make_unique<HitsOptionalParams>(maxIterations, tolerance);
    }
};

HitsOptionalParams::HitsOptionalParams(const expression_vector& optionalParams)
    : MaxIterationOptionalParams{constructMaxIterationParam(optionalParams)} {
    for (auto& optionalParam : optionalParams) {
        auto paramName = StringUtils::getLower(optionalParam->getAlias());
        if (paramName == Tolerance::NAME) {
            tolerance = function::OptionalParam<Tolerance>(optionalParam);
        } else if (paramName == MaxIterations::NAME) {
            continue;
        } else {
            throw BinderException{"Unknown optional parameter: " + optionalParam->getAlias()};
        }
    }
}

struct HitsBindData final : public GDSBindData {
    HitsBindData(expression_vector columns, graph::NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<HitsOptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {
        this->optionalParams = std::move(optionalParams);
    }

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<HitsBindData>(*this);
    }
};

static void addCAS(std::atomic<double>& origin, double valToAdd) {
    auto expected = origin.load(std::memory_order_relaxed);
    auto desired = expected + valToAdd;
    while (!origin.compare_exchange_strong(expected, desired)) {
        desired = expected + valToAdd;
    }
}

// Stores values (hub or authority) for all nodes
class HitsValues {
public:
    HitsValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm, double val) {
        for (const auto& [tableID, maxOffset] : maxOffsetMap) {
            valueMap.allocate(tableID, maxOffset, mm);
            pinTable(tableID);
            for (auto i = 0u; i < maxOffset; ++i) {
                values[i].store(val, std::memory_order_relaxed);
            }
        }
    }

    void pinTable(table_id_t tableID) { values = valueMap.getData(tableID); }

    double getValue(offset_t offset) { return values[offset].load(std::memory_order_relaxed); }

    void addValueCAS(offset_t offset, double val) { addCAS(values[offset], val); }

    void setValue(offset_t offset, double val) {
        values[offset].store(val, std::memory_order_relaxed);
    }

private:
    std::atomic<double>* values = nullptr;
    GDSDenseObjectManager<std::atomic<double>> valueMap;
};

class HitsAuxiliaryState : public GDSAuxiliaryState {
public:
    HitsAuxiliaryState(HitsValues& sourceValues, HitsValues& targetValues)
        : sourceValues{sourceValues}, targetValues{targetValues} {}

    void beginFrontierCompute(table_id_t fromTableID, table_id_t toTableID) override {
        sourceValues.pinTable(toTableID);
        targetValues.pinTable(fromTableID);
    }

    void switchToDense(ExecutionContext*, Graph*) override {}

private:
    HitsValues& sourceValues;
    HitsValues& targetValues;
};

// For authority update: sum hub values of in-neighbors
// For hub update: sum authority values of out-neighbors
class HitsEdgeCompute : public EdgeCompute {
public:
    HitsEdgeCompute(HitsValues& sourceValues, HitsValues& targetValues)
        : sourceValues{sourceValues}, targetValues{targetValues} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, graph::NbrScanState::Chunk& chunk,
        bool) override {
        if (chunk.size() > 0) {
            double valToAdd = 0;
            chunk.forEach([&](auto neighbors, auto, auto i) {
                auto nbrNodeID = neighbors[i];
                valToAdd += sourceValues.getValue(nbrNodeID.offset);
            });
            targetValues.addValueCAS(boundNodeID.offset, valToAdd);
        }
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<HitsEdgeCompute>(sourceValues, targetValues);
    }

private:
    HitsValues& sourceValues;
    HitsValues& targetValues;
};

// Compute sum of squares for L2 normalization
class HitsSumSquaresVertexCompute : public GDSVertexCompute {
public:
    HitsSumSquaresVertexCompute(std::atomic<double>& sumSquares, HitsValues& values,
        NodeOffsetMaskMap* nodeMask)
        : GDSVertexCompute{nodeMask}, sumSquares{sumSquares}, values{values} {}

    void beginOnTableInternal(table_id_t tableID) override { values.pinTable(tableID); }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto val = values.getValue(i);
            addCAS(sumSquares, val * val);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<HitsSumSquaresVertexCompute>(sumSquares, values, nodeMask);
    }

private:
    std::atomic<double>& sumSquares;
    HitsValues& values;
};

// Normalize values by L2 norm
class HitsNormalizeVertexCompute : public GDSVertexCompute {
public:
    HitsNormalizeVertexCompute(double l2Norm, HitsValues& values, NodeOffsetMaskMap* nodeMask)
        : GDSVertexCompute{nodeMask}, l2Norm{l2Norm}, values{values} {}

    void beginOnTableInternal(table_id_t tableID) override { values.pinTable(tableID); }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto val = values.getValue(i);
            auto normalized = (l2Norm > 0) ? val / l2Norm : 0.0;
            values.setValue(i, normalized);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<HitsNormalizeVertexCompute>(l2Norm, values, nodeMask);
    }

private:
    double l2Norm;
    HitsValues& values;
};

// Compute convergence diff between old and new values, and reset old values
class HitsDiffVertexCompute : public GDSVertexCompute {
public:
    HitsDiffVertexCompute(std::atomic<double>& diff, HitsValues& oldValues, HitsValues& newValues,
        NodeOffsetMaskMap* nodeMask)
        : GDSVertexCompute{nodeMask}, diff{diff}, oldValues{oldValues}, newValues{newValues} {}

    void beginOnTableInternal(table_id_t tableID) override {
        oldValues.pinTable(tableID);
        newValues.pinTable(tableID);
    }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto oldVal = oldValues.getValue(i);
            auto newVal = newValues.getValue(i);
            auto delta = (newVal > oldVal) ? (newVal - oldVal) : (oldVal - newVal);
            addCAS(diff, delta);
            // Copy new to old, reset new for next iteration
            oldValues.setValue(i, newVal);
            newValues.setValue(i, 0);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<HitsDiffVertexCompute>(diff, oldValues, newValues, nodeMask);
    }

private:
    std::atomic<double>& diff;
    HitsValues& oldValues;
    HitsValues& newValues;
};

class HitsResultVertexCompute : public GDSResultVertexCompute {
public:
    HitsResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        HitsValues& hubValues, HitsValues& authValues)
        : GDSResultVertexCompute{mm, sharedState}, hubValues{hubValues}, authValues{authValues} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        hubVector = createVector(LogicalType::DOUBLE());
        authorityVector = createVector(LogicalType::DOUBLE());
    }

    void beginOnTableInternal(table_id_t tableID) override {
        hubValues.pinTable(tableID);
        authValues.pinTable(tableID);
    }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            hubVector->setValue<double>(0, hubValues.getValue(i));
            authorityVector->setValue<double>(0, authValues.getValue(i));
            localFT->append(vectors);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<HitsResultVertexCompute>(mm, sharedState, hubValues, authValues);
    }

private:
    HitsValues& hubValues;
    HitsValues& authValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> hubVector;
    std::unique_ptr<ValueVector> authorityVector;
};

static void normalizeValues(const TableFuncInput& input, Graph* graph, HitsValues& values,
    NodeOffsetMaskMap* nodeMask) {
    std::atomic<double> sumSquares;
    sumSquares.store(0.0);
    auto sumSquaresVC = HitsSumSquaresVertexCompute(sumSquares, values, nodeMask);
    GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, sumSquaresVC);

    auto l2Norm = std::sqrt(sumSquares.load());
    auto normalizeVC = HitsNormalizeVertexCompute(l2Norm, values, nodeMask);
    GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, normalizeVC);
}

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    auto clientContext = input.context->clientContext;
    auto transaction = transaction::Transaction::Get(*clientContext);
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto graph = sharedState->graph.get();
    auto maxOffsetMap = graph->getMaxOffsetMap(transaction);
    auto bindData = input.bindData->constPtrCast<HitsBindData>();
    auto& config = bindData->optionalParams->constCast<HitsOptionalParams>();
    auto mm = MemoryManager::Get(*clientContext);
    auto nodeMask = sharedState->getGraphNodeMaskMap();

    // Initialize hub and authority values to 1.0
    auto hub = HitsValues(maxOffsetMap, mm, 1.0);
    auto authority = HitsValues(maxOffsetMap, mm, 1.0);
    // Scratch space for next iteration values
    auto hubNext = HitsValues(maxOffsetMap, mm, 0.0);
    auto authNext = HitsValues(maxOffsetMap, mm, 0.0);

    auto currentIter = 1u;
    auto currentFrontier =
        DenseFrontier::getVisitedFrontier(input.context, graph, nodeMask);
    auto nextFrontier =
        DenseFrontier::getVisitedFrontier(input.context, graph, nodeMask);
    auto frontierPair =
        std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
    auto computeState = GDSComputeState(std::move(frontierPair), nullptr, nullptr);

    while (currentIter <= config.maxIterations.getParamVal()) {
        // Phase 1: Update authority values
        // authority[v] = sum(hub[u] for all neighbors u)
        // Using BOTH direction to handle undirected graphs where edges may be stored one-way
        computeState.frontierPair->resetCurrentIter();
        computeState.frontierPair->setActiveNodesForNextIter();
        computeState.edgeCompute = std::make_unique<HitsEdgeCompute>(hub, authNext);
        computeState.auxiliaryState = std::make_unique<HitsAuxiliaryState>(hub, authNext);
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1);

        // Normalize authority values
        normalizeValues(input, graph, authNext, nodeMask);

        // Phase 2: Update hub values
        // hub[v] = sum(authority[u] for all neighbors u)
        // Using BOTH direction to handle undirected graphs where edges may be stored one-way
        computeState.frontierPair->resetCurrentIter();
        computeState.frontierPair->setActiveNodesForNextIter();
        computeState.edgeCompute = std::make_unique<HitsEdgeCompute>(authNext, hubNext);
        computeState.auxiliaryState = std::make_unique<HitsAuxiliaryState>(authNext, hubNext);
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1);

        // Normalize hub values
        normalizeValues(input, graph, hubNext, nodeMask);

        // Compute convergence for both hub and authority
        std::atomic<double> diff;
        diff.store(0.0);

        auto authDiffVC = HitsDiffVertexCompute(diff, authority, authNext, nodeMask);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, authDiffVC);

        auto hubDiffVC = HitsDiffVertexCompute(diff, hub, hubNext, nodeMask);
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, hubDiffVC);

        // Check convergence
        if (diff.load() < config.tolerance.getParamVal()) {
            break;
        }

        auto progress = static_cast<double>(currentIter) / config.maxIterations.getParamVal();
        ProgressBar::Get(*clientContext)->updateProgress(input.context->queryID, progress);
        currentIter++;
    }

    // Output results
    auto outputVC = std::make_unique<HitsResultVertexCompute>(mm, sharedState, hub, authority);
    GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
    sharedState->factorizedTablePool.mergeLocalTables();
    return 0;
}

static constexpr char HUB_COLUMN_NAME[] = "hub";
static constexpr char AUTHORITY_COLUMN_NAME[] = "authority";

static std::unique_ptr<TableFuncBindData> bindFunc(main::ClientContext* context,
    const TableFuncBindInput* input) {
    auto graphName = input->getLiteralVal<std::string>(0);
    auto graphEntry = GDSFunction::bindGraphEntry(*context, graphName);
    auto nodeOutput = GDSFunction::bindNodeOutput(*input, graphEntry.getNodeEntries());
    expression_vector columns;
    columns.push_back(nodeOutput->constCast<NodeExpression>().getInternalID());
    columns.push_back(input->binder->createVariable(HUB_COLUMN_NAME, LogicalType::DOUBLE()));
    columns.push_back(input->binder->createVariable(AUTHORITY_COLUMN_NAME, LogicalType::DOUBLE()));
    return std::make_unique<HitsBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput, std::make_unique<HitsOptionalParams>(input->optionalParamsLegacy));
}

function_set HitsFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(HitsFunction::name,
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
