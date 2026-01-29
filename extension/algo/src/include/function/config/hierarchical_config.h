#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"

namespace kuzu {
namespace algo_extension {

struct HierarchicalRankSep {
    static constexpr const char* NAME = "ranksep";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 100.0;

    static void validate(double rankSep) {
        if (rankSep <= 0) {
            throw common::BinderException{"Rank separation must be positive."};
        }
    }
};

struct HierarchicalNodeSep {
    static constexpr const char* NAME = "nodesep";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 50.0;

    static void validate(double nodeSep) {
        if (nodeSep <= 0) {
            throw common::BinderException{"Node separation must be positive."};
        }
    }
};

struct HierarchicalIterations {
    static constexpr const char* NAME = "iterations";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::INT64;
    static constexpr int64_t DEFAULT_VALUE = 8;

    static void validate(int64_t iterations) {
        if (iterations < 1) {
            throw common::BinderException{"Iterations must be >= 1."};
        }
    }
};

} // namespace algo_extension
} // namespace kuzu
