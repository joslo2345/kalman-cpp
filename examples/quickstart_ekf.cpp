#include <cmath>
#include <iostream>
#include <kalman/kalman.hpp>
#include <numbers>

// Models are templated on the scalar so the filter can differentiate them:
// no hand-written Jacobians.
struct ConstantVelocity {
    template <typename T>
    Eigen::Matrix<T, 4, 1> predict(const Eigen::Matrix<T, 4, 1>& x, double dt) const {
        Eigen::Matrix<T, 4, 1> out = x;
        out(0) += x(2) * dt;
        out(1) += x(3) * dt;
        return out;
    }
};

struct RangeBearing {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 4, 1>& x) const {
        using std::atan2;  // unqualified calls pick up the autodiff overloads
        using std::sqrt;
        return Eigen::Matrix<T, 2, 1>(sqrt(x(0) * x(0) + x(1) * x(1)), atan2(x(1), x(0)));
    }
    // Optional: wrap the bearing innovation into (-pi, pi].
    Eigen::Vector2d residual(const Eigen::Vector2d& z, const Eigen::Vector2d& z_pred) const {
        Eigen::Vector2d r = z - z_pred;
        r(1) = std::remainder(r(1), 2 * std::numbers::pi);
        return r;
    }
};

int main() {
    kalman::ExtendedKalmanFilter</*N=*/4> ekf(Eigen::Vector4d(10, 5, 0, 0), Eigen::Matrix4d::Identity());
    const Eigen::Matrix4d Q = 1e-3 * Eigen::Matrix4d::Identity();
    const Eigen::Matrix2d R = Eigen::Vector2d(0.1 * 0.1, 0.01 * 0.01).asDiagonal();

    ekf.predict(ConstantVelocity{}, /*dt=*/0.1, Q);
    ekf.update(RangeBearing{}, Eigen::Vector2d(11.2, 0.46), R);  // Jacobians by autodiff

    std::cout << "state " << ekf.state().transpose() << ", NIS " << ekf.nis() << '\n';
}
