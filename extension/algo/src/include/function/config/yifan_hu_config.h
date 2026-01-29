#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"

namespace kuzu {
namespace algo_extension {

struct YHIterations {
    static constexpr const char* NAME = "iterations";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::INT64;
    static constexpr int64_t DEFAULT_VALUE = 100;

    static void validate(int64_t iterations) {
        if (iterations < 1) {
            throw common::BinderException{"Iterations must be >= 1."};
        }
    }
};

struct YHOptimalDistance {
    static constexpr const char* NAME = "k";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 100.0;

    static void validate(double k) {
        if (k <= 0) {
            throw common::BinderException{"Optimal distance K must be positive."};
        }
    }
};

struct YHRelativeStrength {
    static constexpr const char* NAME = "c";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 0.2;

    static void validate(double c) {
        if (c <= 0) {
            throw common::BinderException{"Relative strength C must be positive."};
        }
    }
};

struct YHStepSize {
    static constexpr const char* NAME = "stepsize";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 20.0;

    static void validate(double stepSize) {
        if (stepSize <= 0) {
            throw common::BinderException{"Step size must be positive."};
        }
    }
};

struct YHStepRatio {
    static constexpr const char* NAME = "stepratio";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 0.95;

    static void validate(double stepRatio) {
        if (stepRatio <= 0 || stepRatio >= 1) {
            throw common::BinderException{"Step ratio must be in (0, 1)."};
        }
    }
};

struct YHConvergenceThreshold {
    static constexpr const char* NAME = "convergencethreshold";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 1e-4;

    static void validate(double threshold) {
        if (threshold <= 0) {
            throw common::BinderException{"Convergence threshold must be positive."};
        }
    }
};

} // namespace algo_extension
} // namespace kuzu
