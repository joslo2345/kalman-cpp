#pragma once

#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "scenarios/constant_velocity.hpp"

namespace scenarios {

inline double wrap_angle(double a) {
    return std::remainder(a, 2.0 * std::numbers::pi);
}

// Constant-velocity motion, state [px, py, vx, vy]. Templated so the EKF can
// differentiate it.
struct ConstantVelocity2D {
    template <typename T>
    Eigen::Matrix<T, 4, 1> predict(const Eigen::Matrix<T, 4, 1>& x, double dt) const {
        Eigen::Matrix<T, 4, 1> out = x;
        out(0) += x(2) * dt;
        out(1) += x(3) * dt;
        return out;
    }
};

// Same motion model, with a hand-written Jacobian.
struct ConstantVelocity2DAnalytic : ConstantVelocity2D {
    Eigen::Matrix4d jacobian(const Eigen::Vector4d&, double dt) const { return cv_transition(dt); }
};

// Range and bearing from a sensor at the origin.
struct RangeBearingModel {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 4, 1>& x) const {
        using std::atan2;
        using std::sqrt;
        Eigen::Matrix<T, 2, 1> z;
        z << sqrt(x(0) * x(0) + x(1) * x(1)), atan2(x(1), x(0));
        return z;
    }

    Eigen::Vector2d residual(const Eigen::Vector2d& z, const Eigen::Vector2d& z_pred) const {
        Eigen::Vector2d r = z - z_pred;
        r(1) = wrap_angle(r(1));
        return r;
    }

    Eigen::Matrix<double, 2, 4> analytic_jacobian(const Eigen::Vector4d& x) const {
        const double r2 = x(0) * x(0) + x(1) * x(1);
        const double r = std::sqrt(r2);
        Eigen::Matrix<double, 2, 4> J = Eigen::Matrix<double, 2, 4>::Zero();
        J(0, 0) = x(0) / r;
        J(0, 1) = x(1) / r;
        J(1, 0) = -x(1) / r2;
        J(1, 1) = x(0) / r2;
        return J;
    }
};

struct RangeBearingModelAnalytic : RangeBearingModel {
    Eigen::Matrix<double, 2, 4> jacobian(const Eigen::Vector4d& x) const { return analytic_jacobian(x); }
};

struct RangeBearingScenario {
    double dt;
    Eigen::Matrix4d Q, P0;
    Eigen::Matrix2d R;
    Eigen::Vector4d x0;
    std::vector<Eigen::Vector4d> truth;
    std::vector<Eigen::Vector2d> measurements;
};

// A target that passes 20 m from the sensor, so the measurement model is
// strongly nonlinear near closest approach.
inline RangeBearingScenario range_bearing(int steps, unsigned seed) {
    RangeBearingScenario sc;
    sc.dt = 0.1;
    sc.Q = cv_process_noise(sc.dt, 0.5);
    sc.R = Eigen::Vector2d(0.5 * 0.5, 0.01 * 0.01).asDiagonal();
    sc.x0 << -50.0, 20.0, 5.0, 0.0;
    sc.P0 = Eigen::Vector4d(25.0, 25.0, 4.0, 4.0).asDiagonal();

    std::mt19937 rng(seed);
    const RangeBearingModel h;
    const Eigen::Matrix4d F = cv_transition(sc.dt);
    Eigen::Vector4d x = sc.x0 + sample<4>(rng, sc.P0);
    for (int k = 0; k < steps; ++k) {
        x = F * x + sample<4>(rng, sc.Q);
        Eigen::Vector2d z = h.measure(x) + sample<2>(rng, sc.R);
        z(1) = wrap_angle(z(1));
        sc.truth.push_back(x);
        sc.measurements.push_back(z);
    }
    return sc;
}

// Random states away from the sensor singularity at the origin.
template <int N>
std::vector<Eigen::Matrix<double, N, 1>> random_states(int count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-100.0, 100.0);
    std::vector<Eigen::Matrix<double, N, 1>> out;
    while (static_cast<int>(out.size()) < count) {
        Eigen::Matrix<double, N, 1> x;
        for (int i = 0; i < N; ++i) x(i) = u(rng);
        if (x.template head<2>().norm() > 1.0) out.push_back(x);
    }
    return out;
}

}  // namespace scenarios
