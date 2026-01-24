#include "main/algo_extension.h"

#include "function/algo_function.h"
#include "main/client_context.h"

namespace kuzu {
namespace algo_extension {

using namespace extension;

void AlgoExtension::load(main::ClientContext* context) {
    auto& db = *context->getDatabase();
    ExtensionUtils::addTableFunc<SCCFunction>(db);
    ExtensionUtils::addTableFuncAlias<SCCAliasFunction>(db);
    ExtensionUtils::addTableFunc<SCCKosarajuFunction>(db);
    ExtensionUtils::addTableFuncAlias<SCCKosarajuAliasFunction>(db);
    ExtensionUtils::addTableFunc<WeaklyConnectedComponentsFunction>(db);
    ExtensionUtils::addTableFuncAlias<WeaklyConnectedComponentsAliasFunction>(db);
    ExtensionUtils::addTableFunc<PageRankFunction>(db);
    ExtensionUtils::addTableFuncAlias<PageRankAliasFunction>(db);
    ExtensionUtils::addTableFunc<KCoreDecompositionFunction>(db);
    ExtensionUtils::addTableFuncAlias<KCoreDecompositionAliasFunction>(db);
    ExtensionUtils::addTableFunc<LouvainFunction>(db);
    ExtensionUtils::addTableFunc<SpanningForest>(db);
    ExtensionUtils::addTableFuncAlias<SpanningForestAliasFunction>(db);
    ExtensionUtils::addTableFunc<GraphDensityFunction>(db);
    ExtensionUtils::addTableFuncAlias<GraphDensityAliasFunction>(db);
    ExtensionUtils::addTableFunc<ClusteringCoefficientFunction>(db);
    ExtensionUtils::addTableFuncAlias<ClusteringCoefficientAliasFunction>(db);
    ExtensionUtils::addTableFunc<EigenvectorCentralityFunction>(db);
    ExtensionUtils::addTableFuncAlias<EigenvectorCentralityAliasFunction>(db);
    ExtensionUtils::addTableFunc<HitsFunction>(db);
    ExtensionUtils::addTableFuncAlias<HitsAliasFunction>(db);
    ExtensionUtils::addTableFunc<ClosenessCentralityFunction>(db);
    ExtensionUtils::addTableFuncAlias<ClosenessCentralityAliasFunction>(db);
    ExtensionUtils::addTableFunc<BetweennessCentralityFunction>(db);
    ExtensionUtils::addTableFuncAlias<BetweennessCentralityAliasFunction>(db);
    // Layout algorithms
    ExtensionUtils::addTableFunc<RandomLayoutFunction>(db);
    ExtensionUtils::addTableFuncAlias<RandomLayoutAliasFunction>(db);
    ExtensionUtils::addTableFunc<CircularLayoutFunction>(db);
    ExtensionUtils::addTableFuncAlias<CircularLayoutAliasFunction>(db);
    ExtensionUtils::addTableFunc<FruchtermanReingoldFunction>(db);
    ExtensionUtils::addTableFuncAlias<FruchtermanReingoldAliasFunction>(db);
    ExtensionUtils::addTableFunc<RadialLayoutFunction>(db);
    ExtensionUtils::addTableFuncAlias<RadialLayoutAliasFunction>(db);
    ExtensionUtils::addTableFunc<HierarchicalLayoutFunction>(db);
    ExtensionUtils::addTableFuncAlias<HierarchicalLayoutAliasFunction>(db);
    ExtensionUtils::addTableFunc<YifanHuLayoutFunction>(db);
    ExtensionUtils::addTableFuncAlias<YifanHuLayoutAliasFunction>(db);
    ExtensionUtils::addTableFunc<ForceAtlas2Function>(db);
    ExtensionUtils::addTableFuncAlias<ForceAtlas2AliasFunction>(db);
}

} // namespace algo_extension
} // namespace kuzu

#if defined(BUILD_DYNAMIC_LOAD)
extern "C" {
// Because we link against the static library on windows, we implicitly inherit KUZU_STATIC_DEFINE,
// which cancels out any exporting, so we can't use KUZU_API.
#if defined(_WIN32)
#define INIT_EXPORT __declspec(dllexport)
#else
#define INIT_EXPORT __attribute__((visibility("default")))
#endif
INIT_EXPORT void init(kuzu::main::ClientContext* context) {
    kuzu::algo_extension::AlgoExtension::load(context);
}

INIT_EXPORT const char* name() {
    return kuzu::algo_extension::AlgoExtension::EXTENSION_NAME;
}
}
#endif
