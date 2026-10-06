#pragma once

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <kalman/so3.hpp>

namespace scenarios {

// Orientation plus gyro bias, a 6-DoF product manifold.
template <typename T>
struct AttitudeBias {
    static constexpr int DoF = 6;
    using Scalar = T;

    kalman::SO3<T> R;
    Eigen::Matrix<T, 3, 1> bias = Eigen::Matrix<T, 3, 1>::Zero();

    AttitudeBias boxplus(const Eigen::Matrix<T, 6, 1>& d) const {
        return {R.boxplus(d.template head<3>()), bias + d.template tail<3>()};
    }
    Eigen::Matrix<T, 6, 1> boxminus(const AttitudeBias& other) const {
        Eigen::Matrix<T, 6, 1> out;
        out << R.boxminus(other.R), bias - other.bias;
        return out;
    }
    template <typename U>
    AttitudeBias<U> cast() const {
        return {R.template cast<U>(), bias.template cast<U>()};
    }
};

// Integrates a bias-corrected gyro reading over dt.
struct GyroModel {
    Eigen::Vector3d omega_meas;

    template <typename T>
    AttitudeBias<T> predict(const AttitudeBias<T>& x, double dt) const {
        const Eigen::Matrix<T, 3, 1> w = (omega_meas.template cast<T>() - x.bias) * dt;
        return {{(x.R.q * kalman::so3::exp(w)).normalized()}, x.bias};
    }
};

// A known world-frame direction observed in the body frame (accelerometer
// gravity direction, magnetometer field direction).
struct DirectionModel {
    Eigen::Vector3d ref;

    template <typename T>
    Eigen::Matrix<T, 3, 1> measure(const AttitudeBias<T>& x) const {
        return x.R.q.conjugate() * ref.template cast<T>();
    }
};

struct AttitudeScenario {
    static constexpr double dt = 0.01;  // gyro at 100 Hz
    static constexpr int vector_every = 10;  // accel + mag at 10 Hz

    Eigen::Vector3d gravity_ref{0.0, 0.0, 1.0};
    Eigen::Vector3d mag_ref{0.6, 0.0, 0.8};
    double gyro_sigma = 0.01;  // rad/s per sample
    double accel_sigma = 0.02;
    double mag_sigma = 0.03;

    AttitudeBias<double> truth0;
    std::vector<AttitudeBias<double>> truth;  // after each gyro step
    std::vector<Eigen::Vector3d> gyro;
    std::vector<Eigen::Vector3d> accel, mag;  // valid where k % vector_every == vector_every - 1
};

inline Eigen::Vector3d true_rate(double t) {
    return {0.3 * std::sin(0.5 * t), 0.2 * std::cos(0.3 * t), 0.1};
}

inline AttitudeScenario attitude(double duration, unsigned seed) {
    AttitudeScenario sc;
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, 1.0);
    auto noise3 = [&](double s) { return Eigen::Vector3d(s * n(rng), s * n(rng), s * n(rng)); };

    sc.truth0.R.q = kalman::so3::exp(Eigen::Vector3d(0.3, -0.2, 0.4));  // ~30 deg from identity
    sc.truth0.bias = Eigen::Vector3d(0.02, -0.01, 0.015);

    AttitudeBias<double> x = sc.truth0;
    const int steps = static_cast<int>(duration / sc.dt);
    for (int k = 0; k < steps; ++k) {
        const Eigen::Vector3d w = true_rate((k + 0.5) * sc.dt);
        x.R.q = (x.R.q * kalman::so3::exp(Eigen::Vector3d(w * sc.dt))).normalized();
        sc.truth.push_back(x);
        sc.gyro.push_back(w + x.bias + noise3(sc.gyro_sigma));
        sc.accel.push_back(x.R.q.conjugate() * sc.gravity_ref + noise3(sc.accel_sigma));
        sc.mag.push_back(x.R.q.conjugate() * sc.mag_ref + noise3(sc.mag_sigma));
    }
    return sc;
}

}  // namespace scenarios
