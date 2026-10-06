#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"
#include "scenarios/range_bearing.hpp"

namespace {

constexpr int T = 30;
// Measurements are dropped for these steps to exercise predict-only steps.
bool in_gap(int k) { return k >= 12 && k < 18; }

struct PositionModel {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 4, 1>& x) const {
        return x.template head<2>();
    }
};

// Linear-Gaussian problem over states x_0..x_T with a full-rank Q (the batch
// solution below needs Q^-1).
scenarios::LinearScenario<4, 2> problem(unsigned seed) {
    auto sc = scenarios::constant_velocity_2d(T, seed);
    sc.Q += 0.01 * Eigen::Matrix4d::Identity();
    return sc;
}

// Exact posterior of all states given all measurements, from the information
// form of the whole problem: mean = L^-1 eta, covariance = L^-1.
std::vector<kalman::RtsSmoother<4>::Estimate> batch_solution(const scenarios::LinearScenario<4, 2>& sc) {
    constexpr int D = 4 * (T + 1);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(D, D);
    Eigen::VectorXd eta = Eigen::VectorXd::Zero(D);
    const Eigen::Matrix4d P0i = sc.P0.inverse(), Qi = sc.Q.inverse();
    const Eigen::Matrix2d Ri = sc.R.inverse();

    L.block<4, 4>(0, 0) += P0i;
    eta.segment<4>(0) += P0i * sc.x0;
    for (int k = 1; k <= T; ++k) {
        const int a = 4 * (k - 1), b = 4 * k;
        L.block<4, 4>(b, b) += Qi;
        L.block<4, 4>(a, a) += sc.F.transpose() * Qi * sc.F;
        L.block<4, 4>(a, b) -= sc.F.transpose() * Qi;
        L.block<4, 4>(b, a) -= Qi * sc.F;
        if (!in_gap(k)) {
            L.block<4, 4>(b, b) += sc.H.transpose() * Ri * sc.H;
            eta.segment<4>(b) += sc.H.transpose() * Ri * sc.measurements[k - 1];
        }
    }
    const Eigen::MatrixXd cov = L.inverse();
    const Eigen::VectorXd mean = cov * eta;

    std::vector<kalman::RtsSmoother<4>::Estimate> out(T + 1);
    for (int k = 0; k <= T; ++k) out[k] = {mean.segment<4>(4 * k), cov.block<4, 4>(4 * k, 4 * k)};
    return out;
}

// Runs a filter forward over the problem, recording into a smoother.
template <typename Filter, typename Predict, typename Update>
std::vector<kalman::RtsSmoother<4>::Estimate> smooth_with(Filter& f, const scenarios::LinearScenario<4, 2>& sc,
                                                          Predict predict, Update update) {
    kalman::RtsSmoother<4> smoother;
    smoother.record(f);
    for (int k = 1; k <= T; ++k) {
        predict(f);
        if (!in_gap(k)) REQUIRE(update(f, sc.measurements[k - 1]));
        smoother.record(f);
    }
    return smoother.smooth();
}

void require_close(const std::vector<kalman::RtsSmoother<4>::Estimate>& a,
                   const std::vector<kalman::RtsSmoother<4>::Estimate>& b, double tol) {
    REQUIRE(a.size() == b.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        CAPTURE(k);
        REQUIRE((a[k].x - b[k].x).cwiseAbs().maxCoeff() < tol);
        REQUIRE((a[k].P - b[k].P).cwiseAbs().maxCoeff() < tol);
    }
}

}  // namespace

TEST_CASE("RTS smoother equals the batch solution for every filter", "[unit][smoother]") {
    const auto sc = problem(21);
    const auto exact = batch_solution(sc);
    const scenarios::ConstantVelocity2D cv;
    const PositionModel pos;

    SECTION("linear KF") {
        kalman::LinearFilter<4, 2> f(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
        require_close(smooth_with(f, sc, [](auto& f) { f.predict(); },
                                  [](auto& f, const auto& z) { return f.update(z); }),
                      exact, 1e-9);
    }
    SECTION("EKF") {
        kalman::ExtendedKalmanFilter<4> f(sc.x0, sc.P0);
        require_close(smooth_with(f, sc, [&](auto& f) { f.predict(cv, 0.1, sc.Q); },
                                  [&](auto& f, const auto& z) { return f.update(pos, z, sc.R); }),
                      exact, 1e-9);
    }
    SECTION("UKF") {
        kalman::UnscentedKalmanFilter<4> f(sc.x0, sc.P0);
        require_close(smooth_with(f, sc, [&](auto& f) { REQUIRE(f.predict(cv, 0.1, sc.Q)); },
                                  [&](auto& f, const auto& z) { return f.update(pos, z, sc.R); }),
                      exact, 1e-9);
    }
    SECTION("SR-UKF") {
        kalman::SquareRootUnscentedKalmanFilter<4> f(sc.x0, sc.P0);
        require_close(smooth_with(f, sc, [&](auto& f) { REQUIRE(f.predict(cv, 0.1, sc.Q)); },
                                  [&](auto& f, const auto& z) { return f.update(pos, z, sc.R); }),
                      exact, 1e-9);
    }
}

TEST_CASE("Smoothing lowers the error of the linear KF", "[unit][smoother]") {
    double filt_sq = 0.0, smooth_sq = 0.0;
    for (unsigned seed = 0; seed < 20; ++seed) {
        auto sc = scenarios::constant_velocity_2d(500, seed);
        kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
        kalman::RtsSmoother<4> smoother;
        std::vector<Eigen::Vector4d> filtered;
        for (const auto& z : sc.measurements) {
            kf.predict();
            REQUIRE(kf.update(z));
            smoother.record(kf);
            filtered.push_back(kf.state());
        }
        const auto smoothed = smoother.smooth();
        for (std::size_t k = 0; k < sc.truth.size(); ++k) {
            filt_sq += (filtered[k] - sc.truth[k]).squaredNorm();
            smooth_sq += (smoothed[k].x - sc.truth[k]).squaredNorm();
        }
    }
    UNSCOPED_INFO("[report] CV RMSE filtered " << std::sqrt(filt_sq / 10000) << " smoothed "
                                               << std::sqrt(smooth_sq / 10000));
    REQUIRE(smooth_sq < 0.8 * filt_sq);
}

TEST_CASE("Unscented RTS smoothing lowers the error on a close-pass problem", "[unit][smoother][ukf]") {
    double filt_sq = 0.0, smooth_sq = 0.0;
    for (unsigned seed = 0; seed < 50; ++seed) {
        auto sc = scenarios::range_bearing(100, seed, scenarios::close_pass());
        kalman::UnscentedKalmanFilter<4> ukf(sc.x0, sc.P0);
        kalman::RtsSmoother<4> smoother;
        std::vector<Eigen::Vector4d> filtered;
        for (const auto& z : sc.measurements) {
            REQUIRE(ukf.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q));
            REQUIRE(ukf.update(scenarios::RangeBearingModel{}, z, sc.R));
            smoother.record(ukf);
            filtered.push_back(ukf.state());
        }
        const auto smoothed = smoother.smooth();
        for (std::size_t k = 0; k < sc.truth.size(); ++k) {
            filt_sq += (filtered[k].head<2>() - sc.truth[k].head<2>()).squaredNorm();
            smooth_sq += (smoothed[k].x.head<2>() - sc.truth[k].head<2>()).squaredNorm();
        }
    }
    UNSCOPED_INFO("[report] close-pass position RMSE filtered " << std::sqrt(filt_sq / 5000) << " smoothed "
                                                                << std::sqrt(smooth_sq / 5000));
    REQUIRE(smooth_sq < 0.8 * filt_sq);
}

TEST_CASE("RTS smoother rejects records without exactly one predict", "[unit][smoother]") {
    auto sc = scenarios::constant_velocity_2d(1, 0);
    kalman::LinearFilter<4, 2> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    kalman::RtsSmoother<4> smoother;
    smoother.record(kf);

    SECTION("no predict") {
        REQUIRE(kf.update(sc.measurements[0]));
        REQUIRE_THROWS_AS(smoother.record(kf), std::logic_error);
    }
    SECTION("two predicts") {
        kf.predict();
        kf.predict();
        REQUIRE_THROWS_AS(smoother.record(kf), std::logic_error);
    }
    SECTION("one predict") {
        kf.predict();
        REQUIRE_NOTHROW(smoother.record(kf));
        REQUIRE(smoother.smooth().size() == 2);
    }
}
