// This file must NOT compile: the model has no jacobian() and its predict()
// only accepts double, so the EKF cannot differentiate it.
#include <kalman/kalman.hpp>

struct DoubleOnly {
    Eigen::Vector2d predict(const Eigen::Vector2d& x, double dt) const { return x * dt; }
};

int main() {
    kalman::ExtendedKalmanFilter<2> ekf(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
    ekf.predict(DoubleOnly{}, 0.1, Eigen::Matrix2d::Identity());
}
