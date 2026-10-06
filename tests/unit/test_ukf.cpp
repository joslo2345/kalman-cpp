#include <cmath>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"
#include "scenarios/range_bearing.hpp"

namespace {

struct PositionModel {
    Eigen::Vector2d measure(const Eigen::Vector4d& x) const { return x.head<2>(); }
};

template <typename Filter>
double position_rmse(Filter& f, const scenarios::RangeBearingScenario& sc) {
    double sum = 0.0;
    for (std::size_t k = 0; k < sc.measurements.size(); ++k) {
        f.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q);
        REQUIRE(f.update(scenarios::RangeBearingModel{}, sc.measurements[k], sc.R));
        sum += (f.state().template head<2>() - sc.truth[k].head<2>()).squaredNorm();
    }
    return std::sqrt(sum / static_cast<double>(sc.measurements.size()));
}

template <typename Mat>
bool is_spd(const Mat& P) {
    return Eigen::LLT<Mat>(P).info() == Eigen::Success;
}

}  // namespace

// With kappa = 0 the central sigma-point weight is 1 - 1/alpha^2: 0 for
// alpha = 1, but about -1e6 for alpha = 1e-3. Means and covariances are then
// differences of terms ~1e6 times larger than the result, which costs ~6
// digits; observed worst-case gaps are ~1e-12 and ~7e-7 respectively.
struct AlphaCase {
    double alpha;
    double tol;
};

TEST_CASE("UKF and SR-UKF with linear models match the linear KF", "[unit][ukf]") {
    const auto [alpha, tol] = GENERATE(AlphaCase{1.0, 1e-10}, AlphaCase{1e-3, 1e-5});
    CAPTURE(alpha);
    const kalman::UnscentedParams params{alpha, 2.0, 0.0};

    auto sc = scenarios::constant_velocity_2d(1000, 42);
    kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    kalman::UnscentedKalmanFilter<4> ukf(sc.x0, sc.P0, params);
    kalman::SquareRootUnscentedKalmanFilter<4> srukf(sc.x0, sc.P0, params);

    for (const auto& z : sc.measurements) {
        kf.predict();
        REQUIRE(kf.update(z));
        REQUIRE(ukf.predict(scenarios::ConstantVelocity2D{}, 0.1, sc.Q));
        REQUIRE(ukf.update(PositionModel{}, z, sc.R));
        // sc.Q is rank-deficient, which the SR-UKF must handle.
        REQUIRE(srukf.predict(scenarios::ConstantVelocity2D{}, 0.1, sc.Q));
        REQUIRE(srukf.update(PositionModel{}, z, sc.R));

        REQUIRE((ukf.state() - kf.state()).cwiseAbs().maxCoeff() < tol);
        REQUIRE((ukf.covariance() - kf.covariance()).cwiseAbs().maxCoeff() < tol);
        REQUIRE((srukf.state() - kf.state()).cwiseAbs().maxCoeff() < tol);
        REQUIRE((srukf.covariance() - kf.covariance()).cwiseAbs().maxCoeff() < tol);
        REQUIRE(std::abs(srukf.nis() - kf.nis()) < tol);
    }
}

TEST_CASE("SR-UKF matches the UKF on a nonlinear problem", "[unit][ukf]") {
    // alpha = 1e-3 gives a large negative central weight, which exercises the
    // Cholesky downdate path in the SR-UKF.
    const auto [alpha, tol] = GENERATE(AlphaCase{1.0, 1e-10}, AlphaCase{1e-3, 1e-5});
    CAPTURE(alpha);
    const kalman::UnscentedParams params{alpha, 2.0, 0.0};

    auto sc = scenarios::range_bearing(100, 9, scenarios::close_pass());
    kalman::UnscentedKalmanFilter<4> ukf(sc.x0, sc.P0, params);
    kalman::SquareRootUnscentedKalmanFilter<4> srukf(sc.x0, sc.P0, params);
    for (const auto& z : sc.measurements) {
        REQUIRE(ukf.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q));
        REQUIRE(ukf.update(scenarios::RangeBearingModel{}, z, sc.R));
        REQUIRE(srukf.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q));
        REQUIRE(srukf.update(scenarios::RangeBearingModel{}, z, sc.R));
        REQUIRE((srukf.state() - ukf.state()).cwiseAbs().maxCoeff() < tol);
        REQUIRE((srukf.covariance() - ukf.covariance()).cwiseAbs().maxCoeff() < tol);
    }
}

TEST_CASE("UKF beats the EKF on a close-pass range-bearing problem", "[unit][ukf][ekf]") {
    double ukf_rmse = 0.0, ekf_rmse = 0.0;
    const int seeds = 200;
    for (int seed = 0; seed < seeds; ++seed) {
        auto sc = scenarios::range_bearing(100, seed, scenarios::close_pass());
        kalman::UnscentedKalmanFilter<4> ukf(sc.x0, sc.P0);
        kalman::ExtendedKalmanFilter<4> ekf(sc.x0, sc.P0);
        ukf_rmse += position_rmse(ukf, sc);
        ekf_rmse += position_rmse(ekf, sc);
    }
    UNSCOPED_INFO("[report] close-pass UKF RMSE " << ukf_rmse / seeds << " vs EKF RMSE " << ekf_rmse / seeds);
    // Observed ratio is 0.75-0.86 across disjoint blocks of 200 seeds.
    REQUIRE(ukf_rmse < 0.95 * ekf_rmse);
}

TEST_CASE("Unscented filters average bearings across +/-pi", "[unit][ukf]") {
    // Target just behind the sensor, so the sigma points' bearings straddle the
    // branch cut at +/-pi.
    const Eigen::Vector4d x0(-10.0, 0.01, 0.0, 0.0);
    const Eigen::Matrix4d P0 = Eigen::Vector4d(1.0, 1.0, 1.0, 1.0).asDiagonal();
    const Eigen::Matrix2d R = Eigen::Vector2d(0.01, 0.0001).asDiagonal();
    const Eigen::Vector2d z(10.0, std::atan2(-0.01, -10.0));

    kalman::UnscentedKalmanFilter<4> ukf(x0, P0);
    kalman::SquareRootUnscentedKalmanFilter<4> srukf(x0, P0);
    REQUIRE(ukf.update(scenarios::RangeBearingModel{}, z, R));
    REQUIRE(srukf.update(scenarios::RangeBearingModel{}, z, R));

    for (const Eigen::Vector4d& x : {ukf.state(), srukf.state()}) {
        REQUIRE((x.head<2>() - Eigen::Vector2d(-10.0, -0.01)).norm() < 0.05);
    }
    REQUIRE(ukf.nis() < 10.0);
    REQUIRE(srukf.nis() < 10.0);
}

TEST_CASE("SR-UKF keeps a valid covariance over a long single-precision run", "[unit][ukf]") {
    auto sc = scenarios::constant_velocity_2d(1, 3);
    const Eigen::Matrix4f Q = sc.Q.cast<float>();
    const Eigen::Matrix2f R = sc.R.cast<float>();
    const Eigen::Matrix4f F = sc.F.cast<float>();
    struct CV {
        Eigen::Vector4f predict(const Eigen::Vector4f& x, double dt) const {
            Eigen::Vector4f out = x;
            out.head<2>() += static_cast<float>(dt) * x.tail<2>();
            return out;
        }
    };
    struct Pos {
        Eigen::Vector2f measure(const Eigen::Vector4f& x) const { return x.head<2>(); }
    };

    kalman::SquareRootUnscentedKalmanFilter<4, float> srukf(sc.x0.cast<float>(), sc.P0.cast<float>());
    std::mt19937 rng(3);
    Eigen::Vector4f x = sc.x0.cast<float>();
    for (int k = 0; k < 100'000; ++k) {
        x = F * x;
        REQUIRE(srukf.predict(CV{}, 0.1, Q));
        REQUIRE(srukf.update(Pos{}, Eigen::Vector2f(x.head<2>() + scenarios::sample<2>(rng, R)), R));
        REQUIRE(is_spd(srukf.covariance()));
    }
}

TEST_CASE("Unscented filters reject invalid inputs", "[unit][ukf]") {
    const Eigen::Vector4d x0(10.0, 10.0, 0.0, 0.0);
    const Eigen::Matrix4d indefinite = Eigen::Vector4d(1.0, -1.0, 1.0, 1.0).asDiagonal();
    const Eigen::Matrix2d bad_R = -100.0 * Eigen::Matrix2d::Identity();

    SECTION("bad parameters") {
        REQUIRE_THROWS_AS((kalman::UnscentedKalmanFilter<4>(x0, Eigen::Matrix4d::Identity(), {1.0, 2.0, -4.0})),
                          std::invalid_argument);
    }
    SECTION("UKF with an indefinite covariance") {
        kalman::UnscentedKalmanFilter<4> ukf(x0, indefinite);
        REQUIRE_FALSE(ukf.predict(scenarios::ConstantVelocity2D{}, 0.1, Eigen::Matrix4d::Identity()));
        REQUIRE_FALSE(
            ukf.update(scenarios::RangeBearingModel{}, Eigen::Vector2d(14.0, 0.8), Eigen::Matrix2d::Identity().eval()));
        REQUIRE(ukf.state() == x0);
    }
    SECTION("UKF with a non-positive-definite innovation covariance") {
        kalman::UnscentedKalmanFilter<4> ukf(x0, Eigen::Matrix4d::Identity());
        REQUIRE_FALSE(ukf.update(scenarios::RangeBearingModel{}, Eigen::Vector2d(14.0, 0.8), bad_R));
        REQUIRE(ukf.state() == x0);
    }
    SECTION("SR-UKF with an indefinite initial covariance") {
        REQUIRE_THROWS_AS((kalman::SquareRootUnscentedKalmanFilter<4>(x0, indefinite)), std::invalid_argument);
    }
    SECTION("SR-UKF with a non-positive-definite measurement noise") {
        kalman::SquareRootUnscentedKalmanFilter<4> srukf(x0, Eigen::Matrix4d::Identity());
        REQUIRE_FALSE(srukf.update(scenarios::RangeBearingModel{}, Eigen::Vector2d(14.0, 0.8), bad_R));
        REQUIRE(srukf.state() == x0);
    }
}
