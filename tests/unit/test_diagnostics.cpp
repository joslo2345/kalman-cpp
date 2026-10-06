#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"

using Catch::Matchers::WithinRel;
namespace diag = kalman::diagnostics;

TEST_CASE("Chi-square quantiles match reference values", "[unit][diagnostics]") {
    // Standard table values.
    CHECK_THAT(diag::chi2_quantile(0.95, 1), WithinRel(3.841459, 1e-6));
    CHECK_THAT(diag::chi2_quantile(0.95, 2), WithinRel(5.991465, 1e-6));
    CHECK_THAT(diag::chi2_quantile(0.95, 6), WithinRel(12.59159, 1e-6));
    CHECK_THAT(diag::chi2_quantile(0.99, 10), WithinRel(23.20925, 1e-6));
    CHECK_THAT(diag::chi2_quantile(0.05, 10), WithinRel(3.940299, 1e-6));

    // dof = 2 has the closed form q(p) = -2 ln(1 - p).
    const double p = GENERATE(1e-6, 0.025, 0.5, 0.975, 0.999999);
    CAPTURE(p);
    CHECK_THAT(diag::chi2_quantile(p, 2), WithinRel(-2.0 * std::log1p(-p), 1e-10));
}

TEST_CASE("Chi-square quantile inverts the CDF, including large dof", "[unit][diagnostics]") {
    const double dof = GENERATE(1.0, 4.0, 37.0, 2000.0, 50000.0);
    for (double p : {0.001, 0.025, 0.5, 0.975, 0.999}) {
        CAPTURE(dof, p);
        CHECK_THAT(diag::chi2_cdf(diag::chi2_quantile(p, dof), dof), WithinRel(p, 1e-9));
    }
    // Large dof: close to the normal approximation dof +/- 1.96 sqrt(2 dof).
    if (dof >= 2000) {
        const double z = 1.959964 * std::sqrt(2 * dof);
        CHECK_THAT(diag::chi2_quantile(0.975, dof), WithinRel(dof + z, 2e-3));
        CHECK_THAT(diag::chi2_quantile(0.025, dof), WithinRel(dof - z, 2e-3));
    }
}

namespace {

// 500 runs of the linear KF on the 2-D constant-velocity problem, where the
// truth is simulated with exactly the filter's model. `q_scale` and
// `r_scale` mistune the filter's noise to test that the check notices.
struct Consistency {
    diag::ConsistencyCheck nees, nis;
};

Consistency monte_carlo(double q_scale, double r_scale) {
    constexpr int runs = 500;
    constexpr int steps = 100;
    Consistency c{{4, runs, steps}, {2, runs, steps}};
    for (int run = 0; run < runs; ++run) {
        const auto sc = scenarios::constant_velocity_2d(steps, 1000 + run);
        kalman::LinearFilter<4, 2> kf(sc.F, sc.H, q_scale * sc.Q, r_scale * sc.R, sc.x0, sc.P0);
        for (int k = 0; k < steps; ++k) {
            kf.predict();
            REQUIRE(kf.update(sc.measurements[k]));
            c.nees.add(k, diag::nees(Eigen::Vector4d(sc.truth[k] - kf.state()), kf.covariance()));
            c.nis.add(k, kf.nis());
        }
    }
    return c;
}

}  // namespace

TEST_CASE("Monte Carlo NEES and NIS of a well-tuned KF stay inside the 95% bounds", "[unit][diagnostics]") {
    const auto c = monte_carlo(1.0, 1.0);
    UNSCOPED_INFO("[report] tuned KF: NEES avg " << c.nees.overall_average() << ", " << 100 * c.nees.fraction_inside()
                                                 << "% of steps inside; NIS avg " << c.nis.overall_average() << ", "
                                                 << 100 * c.nis.fraction_inside() << "% inside");
    // Expect ~95% of steps inside. Steps are correlated in time, so allow slack.
    REQUIRE(c.nees.fraction_inside() > 0.85);
    REQUIRE(c.nis.fraction_inside() > 0.85);
    REQUIRE_THAT(c.nees.overall_average(), WithinRel(4.0, 0.05));
    REQUIRE_THAT(c.nis.overall_average(), WithinRel(2.0, 0.05));
}

TEST_CASE("Monte Carlo check flags an overconfident filter", "[unit][diagnostics]") {
    // Filter believes the noise is 4x smaller than it is.
    const auto c = monte_carlo(0.25, 0.25);
    UNSCOPED_INFO("[report] overconfident KF: NEES avg " << c.nees.overall_average() << ", "
                                                         << 100 * c.nees.fraction_inside() << "% of steps inside");
    REQUIRE(c.nees.fraction_inside() < 0.2);
    REQUIRE(c.nis.fraction_inside() < 0.2);
    REQUIRE(c.nees.overall_average() > 8.0);
}

TEST_CASE("nees and nis reject a non-positive-definite covariance", "[unit][diagnostics]") {
    const Eigen::Matrix2d bad = Eigen::Vector2d(1.0, -1.0).asDiagonal();
    REQUIRE_THROWS_AS(diag::nis(Eigen::Vector2d(1.0, 1.0), bad), std::invalid_argument);
    REQUIRE(diag::nees(Eigen::Vector2d(1.0, 2.0), Eigen::Matrix2d(Eigen::Vector2d(1.0, 4.0).asDiagonal())) == 2.0);
}
