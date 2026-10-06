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

struct RangeBearingParams {
    double closest_approach = 20.0;  // m; the target moves along y = closest_approach
    double start_x = -50.0;          // m
    double speed = 5.0;              // m/s, along +x
    double range_sigma = 0.5;        // m
    double bearing_sigma = 0.01;     // rad
    double pos_sigma0 = 5.0;         // m, initial position uncertainty
    double vel_sigma0 = 2.0;         // m/s, initial velocity uncertainty
};

// Default: a target passing 20 m from the sensor with an accurate sensor. The
// model is only mildly nonlinear here, so EKF and UKF perform about the same.
inline RangeBearingScenario range_bearing(int steps, unsigned seed, const RangeBearingParams& p = {}) {
    RangeBearingScenario sc;
    sc.dt = 0.1;
    sc.Q = cv_process_noise(sc.dt, 0.5);
    sc.R = Eigen::Vector2d(p.range_sigma * p.range_sigma, p.bearing_sigma * p.bearing_sigma).asDiagonal();
    sc.x0 << p.start_x, p.closest_approach, p.speed, 0.0;
    const double ps = p.pos_sigma0 * p.pos_sigma0, vs = p.vel_sigma0 * p.vel_sigma0;
    sc.P0 = Eigen::Vector4d(ps, ps, vs, vs).asDiagonal();

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

// Strongly nonlinear: the target passes 2 m from a sensor with an accurate
// range but a poor bearing (0.3 rad), so the posterior is a curved "banana"
// that a first-order linearization represents badly.
inline RangeBearingParams close_pass() {
    RangeBearingParams p;
    p.closest_approach = 2.0;
    p.start_x = -25.0;
    p.range_sigma = 0.2;
    p.bearing_sigma = 0.3;
    return p;
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
