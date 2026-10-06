#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/range_bearing.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using J1 = kalman::Jet<double, 1>;
using J2 = kalman::Jet<double, 2>;

namespace {

// Derivative of a scalar function at x via a 1-input jet.
template <typename F>
double derivative(F f, double x) {
    return f(J1(x, 0)).v(0);
}

}  // namespace

TEST_CASE("Jet derivatives of elementary functions match analytic ones", "[unit][autodiff]") {
    using namespace kalman;
    const double x = GENERATE(0.3, 0.7, 1.9);
    CAPTURE(x);
    // Unqualified calls find the Jet overloads by argument-dependent lookup.
    CHECK_THAT(derivative([](auto t) { return sin(t); }, x), WithinAbs(std::cos(x), 1e-14));
    CHECK_THAT(derivative([](auto t) { return cos(t); }, x), WithinAbs(-std::sin(x), 1e-14));
    CHECK_THAT(derivative([](auto t) { return tan(t); }, x), WithinRel(1.0 / std::pow(std::cos(x), 2), 1e-13));
    CHECK_THAT(derivative([](auto t) { return exp(t); }, x), WithinRel(std::exp(x), 1e-14));
    CHECK_THAT(derivative([](auto t) { return log(t); }, x), WithinRel(1.0 / x, 1e-14));
    CHECK_THAT(derivative([](auto t) { return sqrt(t); }, x), WithinRel(0.5 / std::sqrt(x), 1e-14));
    CHECK_THAT(derivative([](auto t) { return atan(t); }, x), WithinRel(1.0 / (1 + x * x), 1e-14));
    CHECK_THAT(derivative([](auto t) { return tanh(t); }, x), WithinRel(1.0 - std::pow(std::tanh(x), 2), 1e-13));
    CHECK_THAT(derivative([](auto t) { return pow(t, 3.5); }, x), WithinRel(3.5 * std::pow(x, 2.5), 1e-14));
    CHECK_THAT(derivative([](auto t) { return pow(2.0, t); }, x), WithinRel(std::pow(2.0, x) * std::log(2.0), 1e-14));
    CHECK_THAT(derivative([](auto t) { return pow(t, t); }, x),
               WithinRel(std::pow(x, x) * (std::log(x) + 1.0), 1e-14));
    CHECK_THAT(derivative([](auto t) { return 1.0 / t; }, x), WithinRel(-1.0 / (x * x), 1e-14));
    if (x < 1.0) {
        CHECK_THAT(derivative([](auto t) { return asin(t); }, x), WithinRel(1.0 / std::sqrt(1 - x * x), 1e-14));
        CHECK_THAT(derivative([](auto t) { return acos(t); }, x), WithinRel(-1.0 / std::sqrt(1 - x * x), 1e-14));
    }
}

TEST_CASE("Jet gradients of two-input expressions match analytic ones", "[unit][autodiff]") {
    using namespace kalman;
    const double a = 1.3, b = -0.4;
    const J2 x(a, 0), y(b, 1);

    const J2 f = x * y / (x - y) - 3.0 / x + 2 * y;
    // d/dx [xy/(x-y)] = -y^2/(x-y)^2,  d/dy [xy/(x-y)] = x^2/(x-y)^2
    const double d = (a - b) * (a - b);
    CHECK_THAT(f.a, WithinRel(a * b / (a - b) - 3.0 / a + 2 * b, 1e-14));
    CHECK_THAT(f.v(0), WithinRel(-b * b / d + 3.0 / (a * a), 1e-14));
    CHECK_THAT(f.v(1), WithinRel(a * a / d + 2.0, 1e-14));

    const J2 g = atan2(y, x);
    CHECK_THAT(g.v(0), WithinRel(-b / (a * a + b * b), 1e-14));
    CHECK_THAT(g.v(1), WithinRel(a / (a * a + b * b), 1e-14));

    const J2 h = hypot(x, y);
    CHECK_THAT(h.v(0), WithinRel(a / std::hypot(a, b), 1e-14));
    CHECK_THAT(h.v(1), WithinRel(b / std::hypot(a, b), 1e-14));
}

TEST_CASE("Autodiff Jacobian matches analytic Jacobian", "[unit][autodiff]") {
    scenarios::RangeBearingModel model;
    for (const auto& x : scenarios::random_states<4>(/*count=*/100, /*seed=*/3)) {
        Eigen::Matrix<double, 2, 4> auto_J = kalman::jacobian(model, x);
        Eigen::Matrix<double, 2, 4> exact_J = model.analytic_jacobian(x);
        REQUIRE((auto_J - exact_J).cwiseAbs().maxCoeff() < 1e-12);
    }
}

TEST_CASE("Jets work inside Eigen expressions", "[unit][autodiff]") {
    Eigen::Matrix3d A;
    A << 1, 2, 3, 0, -1, 4, 2, 0.5, 1;
    const Eigen::Vector3d x0(0.2, -1.0, 3.0);

    SECTION("double matrix times jet vector gives A as the Jacobian") {
        auto [y, J] = kalman::value_and_jacobian([&](const auto& x) { return A * x; }, x0);
        REQUIRE((y - A * x0).cwiseAbs().maxCoeff() < 1e-14);
        REQUIRE((J - A).cwiseAbs().maxCoeff() < 1e-14);
    }
    SECTION("norm differentiates to x / |x|") {
        auto J = kalman::jacobian(
            [](const auto& x) {
                Eigen::Matrix<typename std::decay_t<decltype(x)>::Scalar, 1, 1> out;
                out << x.norm();
                return out;
            },
            x0);
        REQUIRE((J.transpose() - x0 / x0.norm()).cwiseAbs().maxCoeff() < 1e-14);
    }
}
