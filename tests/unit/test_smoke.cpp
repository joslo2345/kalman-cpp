#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

namespace {
struct ConstantModel {
    Eigen::Vector<double, 2> predict(const Eigen::Vector<double, 2>& x, double) const { return x; }
};
}  // namespace

TEST_CASE("Umbrella header compiles and concepts work", "[unit]") {
    STATIC_REQUIRE(kalman::ProcessModel<ConstantModel, 2>);
    STATIC_REQUIRE_FALSE(kalman::ProcessModel<int, 2>);
}
