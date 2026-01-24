#include "binder/binder.h"
#include "function/algo_function.h"
#include "function/degrees.h"
#include "function/gds/gds.h"
#include "function/gds/gds_utils.h"
#include "function/table/bind_data.h"
#include "function/table/bind_input.h"
#include "graph/on_disk_graph.h"
#include "processor/execution_context.h"
#include "transaction/transaction.h"

using namespace kuzu::binder;
using namespace kuzu::common;
using namespace kuzu::processor;
using namespace kuzu::storage;
using namespace kuzu::graph;
using namespace kuzu::function;

namespace kuzu {
namespace algo_extension {

static constexpr char DENSITY_COLUMN_NAME[] = "density";
static constexpr char NODE_COUNT_COLUMN_NAME[] = "node_count";
static constexpr char EDGE_COUNT_COLUMN_NAME[] = "edge_count";

// Custom bind data that stores the graph entry
struct GraphDensityBindData final : public TableFuncBindData {
    NativeGraphEntry graphEntry;

    GraphDensityBindData(expression_vector columns, NativeGraphEntry graphEntry)
        : TableFuncBindData{std::move(columns), 1 /* numRows - single output row */},
          graphEntry{graphEntry.copy()} {}

    GraphDensityBindData(const GraphDensityBindData& other)
        : TableFuncBindData{other}, graphEntry{other.graphEntry.copy()} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<GraphDensityBindData>(*this);
    }
};

// Custom shared state that holds the graph
struct GraphDensitySharedState final : public TableFuncSharedState {
    std::unique_ptr<Graph> graph;
    bool hasOutput = false;

    explicit GraphDensitySharedState(std::unique_ptr<Graph> graph)
        : TableFuncSharedState{1}, graph{std::move(graph)} {}
};

static std::unique_ptr<TableFuncSharedState> initSharedState(
    const TableFuncInitSharedStateInput& input) {
    auto bindData = input.bindData->constPtrCast<GraphDensityBindData>();
    auto graph =
        std::make_unique<OnDiskGraph>(input.context->clientContext, bindData->graphEntry.copy());
    return std::make_unique<GraphDensitySharedState>(std::move(graph));
}

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput& output) {
    auto sharedState = input.sharedState->ptrCast<GraphDensitySharedState>();

    // Only output once
    {
        std::lock_guard<std::mutex> lock(sharedState->mtx);
        if (sharedState->hasOutput) {
            return 0;
        }
        sharedState->hasOutput = true;
    }

    auto clientContext = input.context->clientContext;
    auto mm = MemoryManager::Get(*clientContext);
    auto graph = sharedState->graph.get();
    auto transaction = transaction::Transaction::Get(*clientContext);

    // Get node count
    auto numNodes = graph->getNumNodes(transaction);

    // Compute degrees to count edges
    auto degrees = Degrees(graph->getMaxOffsetMap(transaction), mm);
    DegreesUtils::computeDegree(input.context, graph, nullptr /* no mask */, &degrees,
        ExtendDirection::BOTH);

    // Sum all degrees (each edge counted twice for undirected with BOTH direction)
    uint64_t totalDegree = 0;
    for (const auto& [tableID, maxOffset] : graph->getMaxOffsetMap(transaction)) {
        degrees.pinTable(tableID);
        for (offset_t i = 0; i < maxOffset; ++i) {
            totalDegree += degrees.getValue(i);
        }
    }

    // For undirected graphs (ExtendDirection::BOTH), each edge is counted twice
    uint64_t numEdges = totalDegree / 2;

    // Calculate density: (2 * edges) / (nodes * (nodes - 1)) for undirected
    // This is equivalent to: totalDegree / (nodes * (nodes - 1))
    double density = 0.0;
    if (numNodes > 1) {
        density = static_cast<double>(totalDegree) /
                  (static_cast<double>(numNodes) * static_cast<double>(numNodes - 1));
    }

    // Output the results
    output.dataChunk.getValueVectorMutable(0).setValue<uint64_t>(0, numNodes);
    output.dataChunk.getValueVectorMutable(1).setValue<uint64_t>(0, numEdges);
    output.dataChunk.getValueVectorMutable(2).setValue<double>(0, density);
    output.dataChunk.state->getSelVectorUnsafe().setSelSize(1);

    return 1;
}

static std::unique_ptr<TableFuncBindData> bindFunc(main::ClientContext* context,
    const TableFuncBindInput* input) {
    auto graphName = input->getLiteralVal<std::string>(0);
    auto graphEntry = GDSFunction::bindGraphEntry(*context, graphName);

    expression_vector columns;
    columns.push_back(
        input->binder->createVariable(NODE_COUNT_COLUMN_NAME, LogicalType::UINT64()));
    columns.push_back(
        input->binder->createVariable(EDGE_COUNT_COLUMN_NAME, LogicalType::UINT64()));
    columns.push_back(input->binder->createVariable(DENSITY_COLUMN_NAME, LogicalType::DOUBLE()));

    return std::make_unique<GraphDensityBindData>(std::move(columns), std::move(graphEntry));
}

function_set GraphDensityFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(name, std::vector{LogicalTypeID::ANY});
    func->bindFunc = bindFunc;
    func->tableFunc = tableFunc;
    func->initSharedStateFunc = initSharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->canParallelFunc = [] { return false; };
    // Use default getLogicalPlanFunc and getPhysicalPlanFunc (not GDS versions)
    result.push_back(std::move(func));
    return result;
}

} // namespace algo_extension
} // namespace kuzu
