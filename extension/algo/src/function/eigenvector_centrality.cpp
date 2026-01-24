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

struct EigenvectorOptionalParams final : public MaxIterationOptionalParams {
    OptionalParam<Tolerance> tolerance;

    explicit EigenvectorOptionalParams(const expression_vector& optionalParams);

    EigenvectorOptionalParams(OptionalParam<MaxIterations> maxIterations,
        OptionalParam<Tolerance> tolerance)
        : MaxIterationOptionalParams{maxIterations}, tolerance{std::move(tolerance)} {}

    void evaluateParams(main::ClientContext* context) override {
        MaxIterationOptionalParams::evaluateParams(context);
        tolerance.evaluateParam(context);
    }

    std::unique_ptr<function::OptionalParams> copy() override {
        return std::make_unique<EigenvectorOptionalParams>(maxIterations, tolerance);
    }
};

EigenvectorOptionalParams::EigenvectorOptionalParams(const expression_vector& optionalParams)
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

struct EigenvectorBindData final : public GDSBindData {
    EigenvectorBindData(expression_vector columns, graph::NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<EigenvectorOptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {
        this->optionalParams = std::move(optionalParams);
    }

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<EigenvectorBindData>(*this);
    }
};

static void addCAS(std::atomic<double>& origin, double valToAdd) {
    auto expected = origin.load(std::memory_order_relaxed);
    auto desired = expected + valToAdd;
    while (!origin.compare_exchange_strong(expected, desired)) {
        desired = expected + valToAdd;
    }
}

// Represents eigenvector centrality values for all nodes
class EValues {
public:
    EValues(table_id_map_t<offset_t> maxOffsetMap, storage::MemoryManager* mm, double val) {
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

class EigenvectorAuxiliaryState : public GDSAuxiliaryState {
public:
    EigenvectorAuxiliaryState(EValues& eCurrent, EValues& eNext)
        : eCurrent{eCurrent}, eNext{eNext} {}

    void beginFrontierCompute(table_id_t fromTableID, table_id_t toTableID) override {
        eCurrent.pinTable(toTableID);
        eNext.pinTable(fromTableID);
    }

    void switchToDense(ExecutionContext*, Graph*) override {}

private:
    EValues& eCurrent;
    EValues& eNext;
};

// Sum the centrality values of all neighbors (no degree division like PageRank)
class ENextUpdateEdgeCompute : public EdgeCompute {
public:
    ENextUpdateEdgeCompute(EValues& eCurrent, EValues& eNext)
        : eCurrent{eCurrent}, eNext{eNext} {}

    std::vector<nodeID_t> edgeCompute(nodeID_t boundNodeID, graph::NbrScanState::Chunk& chunk,
        bool) override {
        if (chunk.size() > 0) {
            double valToAdd = 0;
            chunk.forEach([&](auto neighbors, auto, auto i) {
                auto nbrNodeID = neighbors[i];
                valToAdd += eCurrent.getValue(nbrNodeID.offset);
            });
            eNext.addValueCAS(boundNodeID.offset, valToAdd);
        }
        return {};
    }

    std::unique_ptr<EdgeCompute> copy() override {
        return std::make_unique<ENextUpdateEdgeCompute>(eCurrent, eNext);
    }

private:
    EValues& eCurrent;
    EValues& eNext;
};

// Compute sum of squares for L2 normalization
class ComputeSumSquaresVertexCompute : public GDSVertexCompute {
public:
    ComputeSumSquaresVertexCompute(std::atomic<double>& sumSquares, EValues& eNext,
        NodeOffsetMaskMap* nodeMask)
        : GDSVertexCompute{nodeMask}, sumSquares{sumSquares}, eNext{eNext} {}

    void beginOnTableInternal(table_id_t tableID) override { eNext.pinTable(tableID); }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto val = eNext.getValue(i);
            addCAS(sumSquares, val * val);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<ComputeSumSquaresVertexCompute>(sumSquares, eNext, nodeMask);
    }

private:
    std::atomic<double>& sumSquares;
    EValues& eNext;
};

// Normalize values by L2 norm and compute diff for convergence
class NormalizeAndDiffVertexCompute : public GDSVertexCompute {
public:
    NormalizeAndDiffVertexCompute(double l2Norm, std::atomic<double>& diff, EValues& eCurrent,
        EValues& eNext, NodeOffsetMaskMap* nodeMask)
        : GDSVertexCompute{nodeMask}, l2Norm{l2Norm}, diff{diff}, eCurrent{eCurrent},
          eNext{eNext} {}

    void beginOnTableInternal(table_id_t tableID) override {
        eCurrent.pinTable(tableID);
        eNext.pinTable(tableID);
    }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto next = eNext.getValue(i);
            auto normalized = (l2Norm > 0) ? next / l2Norm : 0.0;
            auto current = eCurrent.getValue(i);
            auto delta = (normalized > current) ? (normalized - current) : (current - normalized);
            addCAS(diff, delta);
            // Store normalized value in eNext, reset eCurrent for next iteration
            eNext.setValue(i, normalized);
            eCurrent.setValue(i, 0);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<NormalizeAndDiffVertexCompute>(l2Norm, diff, eCurrent, eNext,
            nodeMask);
    }

private:
    double l2Norm;
    std::atomic<double>& diff;
    EValues& eCurrent;
    EValues& eNext;
};

class EigenvectorResultVertexCompute : public GDSResultVertexCompute {
public:
    EigenvectorResultVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
        EValues& eValues)
        : GDSResultVertexCompute{mm, sharedState}, eValues{eValues} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        centralityVector = createVector(LogicalType::DOUBLE());
    }

    void beginOnTableInternal(table_id_t tableID) override { eValues.pinTable(tableID); }

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) {
                continue;
            }
            auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            centralityVector->setValue<double>(0, eValues.getValue(i));
            localFT->append(vectors);
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<EigenvectorResultVertexCompute>(mm, sharedState, eValues);
    }

private:
    EValues& eValues;
    std::unique_ptr<ValueVector> nodeIDVector;
    std::unique_ptr<ValueVector> centralityVector;
};

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    auto clientContext = input.context->clientContext;
    auto transaction = transaction::Transaction::Get(*clientContext);
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto graph = sharedState->graph.get();
    auto maxOffsetMap = graph->getMaxOffsetMap(transaction);
    auto bindData = input.bindData->constPtrCast<EigenvectorBindData>();
    auto& config = bindData->optionalParams->constCast<EigenvectorOptionalParams>();
    auto mm = MemoryManager::Get(*clientContext);

    // Initialize all centrality values to 1.0
    auto e1 = EValues(maxOffsetMap, mm, 1.0);
    auto e2 = EValues(maxOffsetMap, mm, 0.0);
    EValues* eCurrent = &e1;
    EValues* eNext = &e2;

    auto currentIter = 1u;
    auto currentFrontier =
        DenseFrontier::getVisitedFrontier(input.context, graph, sharedState->getGraphNodeMaskMap());
    auto nextFrontier =
        DenseFrontier::getVisitedFrontier(input.context, graph, sharedState->getGraphNodeMaskMap());
    auto frontierPair =
        std::make_unique<DenseFrontierPair>(std::move(currentFrontier), std::move(nextFrontier));
    auto computeState = GDSComputeState(std::move(frontierPair), nullptr, nullptr);

    while (currentIter <= config.maxIterations.getParamVal()) {
        computeState.frontierPair->resetCurrentIter();
        computeState.frontierPair->setActiveNodesForNextIter();
        computeState.edgeCompute = std::make_unique<ENextUpdateEdgeCompute>(*eCurrent, *eNext);
        computeState.auxiliaryState = std::make_unique<EigenvectorAuxiliaryState>(*eCurrent, *eNext);

        // Sum neighbor centralities for each node
        GDSUtils::runAlgorithmEdgeCompute(input.context, computeState, graph, ExtendDirection::BOTH,
            1);

        // Compute sum of squares for L2 normalization
        std::atomic<double> sumSquares;
        sumSquares.store(0.0);
        auto sumSquaresVC =
            ComputeSumSquaresVertexCompute(sumSquares, *eNext, sharedState->getGraphNodeMaskMap());
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, sumSquaresVC);

        // Normalize by L2 norm and compute convergence diff
        auto l2Norm = std::sqrt(sumSquares.load());
        std::atomic<double> diff;
        diff.store(0.0);
        auto normalizeVC = NormalizeAndDiffVertexCompute(l2Norm, diff, *eCurrent, *eNext,
            sharedState->getGraphNodeMaskMap());
        GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, normalizeVC);

        // Swap current and next
        std::swap(eCurrent, eNext);

        // Check convergence
        if (diff.load() < config.tolerance.getParamVal()) {
            break;
        }

        auto progress = static_cast<double>(currentIter) / config.maxIterations.getParamVal();
        ProgressBar::Get(*clientContext)->updateProgress(input.context->queryID, progress);
        currentIter++;
    }

    // Output results
    auto outputVC = std::make_unique<EigenvectorResultVertexCompute>(mm, sharedState, *eCurrent);
    GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
    sharedState->factorizedTablePool.mergeLocalTables();
    return 0;
}

static constexpr char CENTRALITY_COLUMN_NAME[] = "centrality";

static std::unique_ptr<TableFuncBindData> bindFunc(main::ClientContext* context,
    const TableFuncBindInput* input) {
    auto graphName = input->getLiteralVal<std::string>(0);
    auto graphEntry = GDSFunction::bindGraphEntry(*context, graphName);
    auto nodeOutput = GDSFunction::bindNodeOutput(*input, graphEntry.getNodeEntries());
    expression_vector columns;
    columns.push_back(nodeOutput->constCast<NodeExpression>().getInternalID());
    columns.push_back(
        input->binder->createVariable(CENTRALITY_COLUMN_NAME, LogicalType::DOUBLE()));
    return std::make_unique<EigenvectorBindData>(std::move(columns), std::move(graphEntry),
        nodeOutput, std::make_unique<EigenvectorOptionalParams>(input->optionalParamsLegacy));
}

function_set EigenvectorCentralityFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(EigenvectorCentralityFunction::name,
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
