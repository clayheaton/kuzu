#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"

namespace kuzu {
namespace algo_extension {

struct FA2Iterations {
    static constexpr const char* NAME = "iterations";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::INT64;
    static constexpr int64_t DEFAULT_VALUE = 100;

    static void validate(int64_t iterations) {
        if (iterations < 1) {
            throw common::BinderException{"Iterations must be >= 1."};
        }
    }
};

struct FA2ScalingRatio {
    static constexpr const char* NAME = "scalingratio";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 2.0;

    static void validate(double scalingRatio) {
        if (scalingRatio <= 0) {
            throw common::BinderException{"Scaling ratio must be positive."};
        }
    }
};

struct FA2Gravity {
    static constexpr const char* NAME = "gravity";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 1.0;
};

struct FA2JitterTolerance {
    static constexpr const char* NAME = "jittertolerance";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 1.0;

    static void validate(double jitterTolerance) {
        if (jitterTolerance <= 0) {
            throw common::BinderException{"Jitter tolerance must be positive."};
        }
    }
};

} // namespace algo_extension
} // namespace kuzu
