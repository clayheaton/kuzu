#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"

namespace kuzu {
namespace algo_extension {

struct FRIterations {
    static constexpr const char* NAME = "iterations";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::INT64;
    static constexpr int64_t DEFAULT_VALUE = 500;

    static void validate(int64_t iterations) {
        if (iterations < 1) {
            throw common::BinderException{"Iterations must be >= 1."};
        }
    }
};

struct FRArea {
    static constexpr const char* NAME = "area";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 10000.0;

    static void validate(double area) {
        if (area <= 0) {
            throw common::BinderException{"Area must be positive."};
        }
    }
};

struct FRGravity {
    static constexpr const char* NAME = "gravity";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 1.0;
};

struct FRSpeed {
    static constexpr const char* NAME = "speed";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 1.0;

    static void validate(double speed) {
        if (speed <= 0) {
            throw common::BinderException{"Speed must be positive."};
        }
    }
};

} // namespace algo_extension
} // namespace kuzu
