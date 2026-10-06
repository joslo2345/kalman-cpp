#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/opencv_helpers.hpp"
#include "scenarios/range_bearing.hpp"

namespace {

using Track = std::vector<Eigen::Vector4d>;

// An EKF built on cv::KalmanFilter: the measurement matrix is re-linearized
// at every step, and correct() gets the pseudo-measurement z - h(x) + H x, so
// that its linear residual z' - H x equals the EKF innovation z - h(x).
Track run_opencv_ekf(const scenarios::RangeBearingScenario& sc) {
    const scenarios::RangeBearingModel h;
    cv::KalmanFilter kf(4, 2, 0, CV_64F);
    cv::eigen2cv(Eigen::Matrix4d(scenarios::cv_transition(sc.dt)), kf.transitionMatrix);
    cv::eigen2cv(sc.Q, kf.processNoiseCov);
    cv::eigen2cv(sc.R, kf.measurementNoiseCov);
    cv::eigen2cv(sc.P0, kf.errorCovPost);
    cv::eigen2cv(sc.x0, kf.statePost);

    Track out;
    for (const auto& z : sc.measurements) {
        const Eigen::Vector4d x_pred = scenarios::from_cv<double, 4>(kf.predict());
        const Eigen::Matrix<double, 2, 4> H = h.analytic_jacobian(x_pred);
        cv::eigen2cv(H, kf.measurementMatrix);
        const Eigen::Vector2d pseudo = h.residual(z, h.measure(x_pred)) + H * x_pred;
        out.push_back(scenarios::from_cv<double, 4>(kf.correct(scenarios::to_cv(pseudo))));
    }
    return out;
}

template <typename Filter>
Track run_ours(const scenarios::RangeBearingScenario& sc) {
    Filter f(sc.x0, sc.P0);
    Track out;
    for (const auto& z : sc.measurements) {
        f.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q);
        REQUIRE(f.update(scenarios::RangeBearingModel{}, z, sc.R));
        out.push_back(f.state());
    }
    return out;
}

double position_rmse(const Track& est, const std::vector<Eigen::Vector4d>& truth) {
    double sum = 0.0;
    for (std::size_t k = 0; k < est.size(); ++k) sum += (est[k].head<2>() - truth[k].head<2>()).squaredNorm();
    return std::sqrt(sum / static_cast<double>(est.size()));
}

}  // namespace

TEST_CASE("The OpenCV-based EKF is a faithful EKF", "[comparison]") {
    // Guards the baseline: it must agree with our EKF, or the comparison
    // below would be against a broken implementation.
    const auto sc = scenarios::range_bearing(100, 3, scenarios::close_pass());
    const Track theirs = run_opencv_ekf(sc);
    const Track ours = run_ours<kalman::ExtendedKalmanFilter<4>>(sc);
    for (std::size_t k = 0; k < ours.size(); ++k) {
        REQUIRE((ours[k] - theirs[k]).cwiseAbs().maxCoeff() < 1e-8);
    }
}

TEST_CASE("UKF beats an OpenCV-based EKF on range-bearing tracking", "[comparison]") {
    double ours_rmse = 0, opencv_ekf_rmse = 0;
    for (int seed = 0; seed < 200; ++seed) {
        auto sc = scenarios::range_bearing(/*steps=*/100, seed, scenarios::close_pass());
        ours_rmse += position_rmse(run_ours<kalman::UnscentedKalmanFilter<4>>(sc), sc.truth);
        opencv_ekf_rmse += position_rmse(run_opencv_ekf(sc), sc.truth);
    }
    WARN("[report] UKF RMSE " << ours_rmse / 200 << " vs OpenCV EKF RMSE " << opencv_ekf_rmse / 200);
    REQUIRE(ours_rmse < opencv_ekf_rmse);
}
