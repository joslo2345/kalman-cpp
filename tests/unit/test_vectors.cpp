#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/range_bearing.hpp"
#include "scenarios/vectors.hpp"

namespace {

// Checks the time-averaged NEES of the first K states (their marginal), which
// is close to K when the truth is simulated with exactly the filter's model.
// For S5 only position and velocity are checked: the gyro bias barely moves
// and is weakly observable, so its error stays near its single initial draw
// and averaging over time cannot bring its NEES towards the mean.
template <int K, int N, int M>
void check_linear(const scenarios::Scenario<N, M>& sc, std::size_t steps) {
    REQUIRE(sc.zs.size() == steps);
    REQUIRE(sc.truth.size() == steps);
    kalman::LinearFilter<N, M> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    double nees = 0.0;
    for (std::size_t k = 0; k < steps; ++k) {
        kf.predict();
        REQUIRE(kf.update(sc.zs[k]));
        const Eigen::Matrix<double, N, 1> e = sc.truth[k] - kf.state();
        nees += kalman::diagnostics::nees(Eigen::Matrix<double, K, 1>(e.template head<K>()),
                                          Eigen::Matrix<double, K, K>(kf.covariance().template topLeftCorner<K, K>()));
    }
    UNSCOPED_INFO("[report] average NEES of the first " << K << " states: " << nees / steps);
    REQUIRE(std::abs(nees / steps - K) < 0.15 * K);
}

}  // namespace

TEST_CASE("Frozen vectors S1, S2 and S5 load and are consistent with their models", "[unit][vectors]") {
    check_linear<2>(scenarios::load<2, 1>("S1"), 10'000);
    check_linear<4>(scenarios::load<4, 2>("S2"), 10'000);
    check_linear<6>(scenarios::load<15, 6>("S5"), 10'000);
}

TEST_CASE("Frozen vector S4 loads in single precision", "[unit][vectors]") {
    const auto sc = scenarios::load<4, 2, float>("S4");
    REQUIRE(sc.zs.size() == 1'000'000);
    REQUIRE(sc.truth.empty());
    REQUIRE(sc.R(0, 0) == 1e-6f);
    REQUIRE(sc.P0(0, 0) == 1e6f);
}

TEST_CASE("Frozen vector S3 shows the UKF beating the EKF", "[unit][vectors][ukf]") {
    const auto sc = scenarios::load_range_bearing();
    REQUIRE(sc.zs.size() == 200);
    REQUIRE(sc.zs[0].size() == 500);
    double ekf_sq = 0.0, ukf_sq = 0.0;
    for (std::size_t s = 0; s < sc.zs.size(); ++s) {
        kalman::ExtendedKalmanFilter<4> ekf(sc.x0, sc.P0);
        kalman::UnscentedKalmanFilter<4> ukf(sc.x0, sc.P0);
        for (std::size_t k = 0; k < sc.zs[s].size(); ++k) {
            ekf.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q);
            ukf.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q);
            REQUIRE(ekf.update(scenarios::RangeBearingModel{}, sc.zs[s][k], sc.R));
            REQUIRE(ukf.update(scenarios::RangeBearingModel{}, sc.zs[s][k], sc.R));
            ekf_sq += (ekf.state().head<2>() - sc.truth[s][k].head<2>()).squaredNorm();
            ukf_sq += (ukf.state().head<2>() - sc.truth[s][k].head<2>()).squaredNorm();
        }
    }
    UNSCOPED_INFO("[report] S3 position RMSE: UKF " << std::sqrt(ukf_sq / 100'000) << ", EKF "
                                                    << std::sqrt(ekf_sq / 100'000));
    REQUIRE(ukf_sq < ekf_sq);
}
