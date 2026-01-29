#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"

namespace kuzu {
namespace algo_extension {

struct RadialMinRadius {
    static constexpr const char* NAME = "minradius";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 50.0;

    static void validate(double minRadius) {
        if (minRadius < 0) {
            throw common::BinderException{"Min radius must be non-negative."};
        }
    }
};

struct RadialRadiusIncrement {
    static constexpr const char* NAME = "radiusincrement";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 100.0;

    static void validate(double radiusIncrement) {
        if (radiusIncrement <= 0) {
            throw common::BinderException{"Radius increment must be positive."};
        }
    }
};

} // namespace algo_extension
} // namespace kuzu
