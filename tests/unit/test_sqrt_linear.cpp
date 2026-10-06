#include <cmath>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"
#include "scenarios/ill_conditioned.hpp"

TEST_CASE("Square-root KF matches the Joseph-form KF in double precision", "[unit][linear][sqrt]") {
    auto sc = scenarios::constant_velocity_2d(1000, 42);
    kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    kalman::SquareRootLinearFilter<4, 2> sr(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    kalman::RtsSmoother<4> kf_smoother, sr_smoother;
    kf_smoother.record(kf);
    sr_smoother.record(sr);
    for (const auto& z : sc.measurements) {
        kf.predict();
        sr.predict();
        REQUIRE(kf.update(z));
        REQUIRE(sr.update(z));
        REQUIRE((sr.state() - kf.state()).cwiseAbs().maxCoeff() < 1e-9);
        REQUIRE((sr.covariance() - kf.covariance()).cwiseAbs().maxCoeff() < 1e-9);
        REQUIRE(std::abs(sr.nis() - kf.nis()) < 1e-9);
        kf_smoother.record(kf);
        sr_smoother.record(sr);
    }
    const auto a = kf_smoother.smooth();
    const auto b = sr_smoother.smooth();
    for (std::size_t k = 0; k < a.size(); ++k) {
        REQUIRE((a[k].x - b[k].x).cwiseAbs().maxCoeff() < 1e-9);
        REQUIRE((a[k].P - b[k].P).cwiseAbs().maxCoeff() < 1e-9);
    }
}

TEST_CASE("Square-root KF keeps precise variances next to vague ones in float", "[unit][linear][sqrt]") {
    // Prior variance 1e6 against measurement variance 1e-6 (sigma 1e-3). After
    // two position measurements with dt = 1, the exact posterior per axis is
    // pos var = sigma^2, vel var = 2 sigma^2, cov = sigma^2: eigenvalues
    // (3 -/+ sqrt(5)) / 2 * sigma^2. The covariance form loses this in float,
    // because 1e-6 added to 1e4 rounds away.
    auto sc = scenarios::ill_conditioned<float>(7);
    kalman::SquareRootLinearFilter<4, 2, float> sr(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    for (long k = 0; k < 2; ++k) {
        sr.predict();
        REQUIRE(sr.update(sc.measure(k)));
    }
    const Eigen::Matrix4f P = sr.covariance();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> es(P);
    const double small = (3 - std::sqrt(5.0)) / 2 * 1e-6, large = (3 + std::sqrt(5.0)) / 2 * 1e-6;
    REQUIRE(std::abs(es.eigenvalues()(0) - small) < 0.05 * small);
    REQUIRE(std::abs(es.eigenvalues()(3) - large) < 0.05 * large);
    REQUIRE(std::abs(P(2, 2) - 2e-6) < 0.05 * 2e-6);
}

TEST_CASE("Square-root KF rejects invalid noise and prior", "[unit][linear][sqrt]") {
    auto sc = scenarios::constant_velocity_2d(1, 0);
    const Eigen::Matrix2d bad_R = -Eigen::Matrix2d::Identity();
    const Eigen::Matrix4d bad_P0 = Eigen::Vector4d(1.0, -1.0, 1.0, 1.0).asDiagonal();
    REQUIRE_THROWS_AS((kalman::SquareRootLinearFilter<4, 2>(sc.F, sc.H, sc.Q, bad_R, sc.x0, sc.P0)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((kalman::SquareRootLinearFilter<4, 2>(sc.F, sc.H, sc.Q, sc.R, sc.x0, bad_P0)),
                      std::invalid_argument);
    // Q may be singular (the constant-velocity Q is rank 2).
    REQUIRE_NOTHROW(kalman::SquareRootLinearFilter<4, 2>(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0));
}
