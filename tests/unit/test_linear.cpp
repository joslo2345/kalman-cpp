#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

template <typename Mat>
bool is_spd(const Mat& P, typename Mat::Scalar sym_tol) {
    if ((P - P.transpose()).cwiseAbs().maxCoeff() > sym_tol) return false;
    return Eigen::LLT<Mat>(P).info() == Eigen::Success;
}

// Textbook filter with the short-form covariance update P = (I - K H) P.
struct NaiveFilter {
    Eigen::Matrix4d F, Q, P;
    Eigen::Matrix<double, 2, 4> H;
    Eigen::Matrix2d R;
    Eigen::Vector4d x;

    void predict() {
        x = F * x;
        P = F * P * F.transpose() + Q;
    }
    void update(const Eigen::Vector2d& z) {
        Eigen::Matrix2d S = H * P * H.transpose() + R;
        Eigen::Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();
        x += K * (z - H * x);
        P = (Eigen::Matrix4d::Identity() - K * H) * P;
    }
};

}  // namespace

TEST_CASE("Scalar KF on a constant matches the closed-form recursive mean", "[unit][linear]") {
    // With F = H = 1 and Q = 0, the posterior is P_k = 1 / (1/P0 + k/R) and
    // x_k = P_k (x0/P0 + sum(z)/R).
    const double P0 = 4.0, R = 0.25, x0 = 1.0;
    kalman::LinearFilter<1, 1> kf(Eigen::Matrix<double, 1, 1>(1.0), Eigen::Matrix<double, 1, 1>(1.0),
                                  Eigen::Matrix<double, 1, 1>(0.0), Eigen::Matrix<double, 1, 1>(R),
                                  Eigen::Matrix<double, 1, 1>(x0), Eigen::Matrix<double, 1, 1>(P0));

    const double zs[] = {2.1, 1.9, 2.3, 1.7, 2.0, 2.2, 1.8};
    double sum = 0.0;
    int k = 0;
    for (double z : zs) {
        kf.predict();
        REQUIRE(kf.update(Eigen::Matrix<double, 1, 1>(z)));
        sum += z;
        ++k;
        const double P = 1.0 / (1.0 / P0 + k / R);
        REQUIRE_THAT(kf.covariance()(0, 0), WithinRel(P, 1e-12));
        REQUIRE_THAT(kf.state()(0), WithinRel(P * (x0 / P0 + sum / R), 1e-12));
    }
}

TEST_CASE("Joseph form matches the textbook filter in double precision", "[unit][linear]") {
    auto sc = scenarios::constant_velocity_2d(1000, 42);
    kalman::LinearFilter<4, 2> ours(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    NaiveFilter naive{sc.F, sc.Q, sc.P0, sc.H, sc.R, sc.x0};

    for (const auto& z : sc.measurements) {
        ours.predict();
        REQUIRE(ours.update(z));
        naive.predict();
        naive.update(z);
        REQUIRE((ours.state() - naive.x).cwiseAbs().maxCoeff() < 1e-9);
        REQUIRE((ours.covariance() - naive.P).cwiseAbs().maxCoeff() < 1e-9);
    }
}

TEST_CASE("Filter tracks the truth and keeps covariance SPD", "[unit][linear]") {
    auto sc = scenarios::constant_velocity_2d(2000, 7);
    kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);

    double sq_err = 0.0, sq_meas_err = 0.0;
    for (std::size_t k = 0; k < sc.measurements.size(); ++k) {
        kf.predict();
        REQUIRE(kf.update(sc.measurements[k]));
        REQUIRE(is_spd(kf.covariance(), 1e-12));
        sq_err += (kf.state().head<2>() - sc.truth[k].head<2>()).squaredNorm();
        sq_meas_err += (sc.measurements[k] - sc.truth[k].head<2>()).squaredNorm();
    }
    // Filtering should beat using the raw measurements as position estimates.
    REQUIRE(sq_err < 0.5 * sq_meas_err);
}

TEST_CASE("NIS equals y^T S^-1 y and averages to the measurement dimension", "[unit][linear]") {
    auto sc = scenarios::constant_velocity_2d(5000, 11);
    kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);

    double nis_sum = 0.0;
    for (const auto& z : sc.measurements) {
        kf.predict();
        const Eigen::Matrix2d S = sc.H * kf.covariance() * sc.H.transpose() + sc.R;
        const Eigen::Vector2d y = z - sc.H * kf.state();
        REQUIRE(kf.update(z));
        REQUIRE_THAT(kf.nis(), WithinRel(y.dot(S.inverse() * y), 1e-9));
        nis_sum += kf.nis();
    }
    // E[NIS] = M = 2 for a consistent filter. With 5000 samples the standard
    // error of the mean is 2/sqrt(5000) ~= 0.028, so 0.15 is a ~5-sigma band.
    REQUIRE_THAT(nis_sum / static_cast<double>(sc.measurements.size()), WithinAbs(2.0, 0.15));
}

TEST_CASE("Update rejects a non-positive-definite innovation covariance", "[unit][linear]") {
    auto sc = scenarios::constant_velocity_2d(1, 0);
    kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    const Eigen::Vector4d x_before = kf.state();
    const Eigen::Matrix2d bad_R = -100.0 * Eigen::Matrix2d::Identity();
    REQUIRE_FALSE(kf.update(sc.measurements[0], bad_R));
    REQUIRE(kf.state() == x_before);
}

TEST_CASE("Single-precision filter stays SPD over a long run", "[unit][linear]") {
    auto sc = scenarios::constant_velocity_2d(1, 3);
    kalman::LinearFilter<4, 2, float> kf(sc.F.cast<float>(), sc.H.cast<float>(), sc.Q.cast<float>(), sc.R.cast<float>(),
                                         sc.x0.cast<float>(), sc.P0.cast<float>());
    std::mt19937 rng(3);
    const Eigen::Matrix2f R = sc.R.cast<float>();
    Eigen::Vector4f x = sc.x0.cast<float>();
    for (int k = 0; k < 100'000; ++k) {
        x = kf.transition() * x;
        kf.predict();
        REQUIRE(kf.update(sc.H.cast<float>() * x + scenarios::sample<2>(rng, R)));
        REQUIRE(is_spd(kf.covariance(), 1e-4f));
    }
}
