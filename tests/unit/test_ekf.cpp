#include <cmath>
#include <numbers>

#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"
#include "scenarios/range_bearing.hpp"

namespace {

struct PositionModel {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 4, 1>& x) const {
        return x.template head<2>();
    }
};

double position_rmse(const std::vector<Eigen::Vector4d>& est, const std::vector<Eigen::Vector4d>& truth) {
    double sum = 0.0;
    for (std::size_t k = 0; k < est.size(); ++k) sum += (est[k].head<2>() - truth[k].head<2>()).squaredNorm();
    return std::sqrt(sum / est.size());
}

template <typename Process, typename Measurement>
std::vector<Eigen::Vector4d> run_ekf(const scenarios::RangeBearingScenario& sc) {
    kalman::ExtendedKalmanFilter<4> ekf(sc.x0, sc.P0);
    std::vector<Eigen::Vector4d> est;
    for (const auto& z : sc.measurements) {
        ekf.predict(Process{}, sc.dt, sc.Q);
        REQUIRE(ekf.update(Measurement{}, z, sc.R));
        est.push_back(ekf.state());
    }
    return est;
}

}  // namespace

TEST_CASE("EKF with linear models matches the linear KF", "[unit][ekf]") {
    auto sc = scenarios::constant_velocity_2d(1000, 42);
    kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    kalman::ExtendedKalmanFilter<4> ekf(sc.x0, sc.P0);

    for (const auto& z : sc.measurements) {
        kf.predict();
        REQUIRE(kf.update(z));
        ekf.predict(scenarios::ConstantVelocity2D{}, 0.1, sc.Q);
        REQUIRE(ekf.update(PositionModel{}, z, sc.R));

        REQUIRE((ekf.state() - kf.state()).cwiseAbs().maxCoeff() < 1e-9);
        REQUIRE((ekf.covariance() - kf.covariance()).cwiseAbs().maxCoeff() < 1e-9);
        REQUIRE(std::abs(ekf.nis() - kf.nis()) < 1e-9);
    }
}

TEST_CASE("Autodiff and analytic Jacobians give the same EKF estimates", "[unit][ekf][autodiff]") {
    auto sc = scenarios::range_bearing(500, 5);
    auto with_autodiff = run_ekf<scenarios::ConstantVelocity2D, scenarios::RangeBearingModel>(sc);
    auto with_analytic =
        run_ekf<scenarios::ConstantVelocity2DAnalytic, scenarios::RangeBearingModelAnalytic>(sc);
    for (std::size_t k = 0; k < with_autodiff.size(); ++k) {
        REQUIRE((with_autodiff[k] - with_analytic[k]).cwiseAbs().maxCoeff() < 1e-10);
    }
}

TEST_CASE("EKF beats polar-to-Cartesian conversion on range-bearing tracking", "[unit][ekf]") {
    double ekf_rmse = 0.0, raw_rmse = 0.0;
    const int seeds = 50;
    for (int seed = 0; seed < seeds; ++seed) {
        auto sc = scenarios::range_bearing(500, seed);
        ekf_rmse += position_rmse(run_ekf<scenarios::ConstantVelocity2D, scenarios::RangeBearingModel>(sc), sc.truth);

        std::vector<Eigen::Vector4d> raw;
        for (const auto& z : sc.measurements) {
            raw.emplace_back(z(0) * std::cos(z(1)), z(0) * std::sin(z(1)), 0.0, 0.0);
        }
        raw_rmse += position_rmse(raw, sc.truth);
    }
    UNSCOPED_INFO("[report] EKF RMSE " << ekf_rmse / seeds << " vs raw RMSE " << raw_rmse / seeds);
    REQUIRE(ekf_rmse < 0.6 * raw_rmse);
}

TEST_CASE("Bearing residual wraps across +/-pi", "[unit][ekf]") {
    // Target just behind the sensor: bearing is close to +pi.
    const Eigen::Vector4d x0(-10.0, 0.01, 0.0, 0.0);
    const Eigen::Matrix4d P0 = Eigen::Vector4d(1.0, 1.0, 1.0, 1.0).asDiagonal();
    const Eigen::Matrix2d R = Eigen::Vector2d(0.01, 0.0001).asDiagonal();
    kalman::ExtendedKalmanFilter<4> ekf(x0, P0);

    // Same physical direction, reported on the other side of the branch cut.
    const double bearing = std::atan2(-0.01, -10.0);
    REQUIRE(bearing < 0.0);
    REQUIRE(ekf.update(scenarios::RangeBearingModel{}, Eigen::Vector2d(10.0, bearing), R));

    // Without wrapping, the ~2*pi innovation would throw the estimate far off.
    REQUIRE((ekf.state().head<2>() - Eigen::Vector2d(-10.0, -0.01)).norm() < 0.05);
    REQUIRE(ekf.nis() < 10.0);
}

TEST_CASE("EKF update rejects a non-positive-definite innovation covariance", "[unit][ekf]") {
    kalman::ExtendedKalmanFilter<4> ekf(Eigen::Vector4d(10, 10, 0, 0), Eigen::Matrix4d::Identity());
    const Eigen::Vector4d before = ekf.state();
    REQUIRE_FALSE(ekf.update(scenarios::RangeBearingModel{}, Eigen::Vector2d(14.0, 0.8),
                             Eigen::Matrix2d(-100.0 * Eigen::Matrix2d::Identity())));
    REQUIRE(ekf.state() == before);
}
