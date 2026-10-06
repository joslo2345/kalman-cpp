#pragma once

#include <random>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

namespace scenarios {

template <int N, int M, typename Scalar = double>
struct LinearScenario {
    Eigen::Matrix<Scalar, N, N> F, Q, P0;
    Eigen::Matrix<Scalar, M, N> H;
    Eigen::Matrix<Scalar, M, M> R;
    Eigen::Matrix<Scalar, N, 1> x0;
    std::vector<Eigen::Matrix<Scalar, N, 1>> truth;
    std::vector<Eigen::Matrix<Scalar, M, 1>> measurements;
};

// Draws a sample from N(0, cov) using a fixed-seed engine.
template <int K, typename Scalar>
Eigen::Matrix<Scalar, K, 1> sample(std::mt19937& rng, const Eigen::Matrix<Scalar, K, K>& cov) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::Matrix<Scalar, K, 1> w;
    for (int i = 0; i < K; ++i) w(i) = static_cast<Scalar>(nd(rng));
    return Eigen::LLT<Eigen::Matrix<Scalar, K, K>>(cov).matrixL() * w;
}

// 2D constant-velocity target, state [px, py, vx, vy], position measurements.
// Note: std::normal_distribution is implementation-defined, so the exact samples
// differ between standard libraries. Cross-library comparisons must use the
// frozen files in tests/vectors/ instead.
inline LinearScenario<4, 2> constant_velocity_2d(int steps, unsigned seed, double dt = 0.1,
                                                 double accel_sigma = 0.5, double meas_sigma = 1.0) {
    LinearScenario<4, 2> sc;
    sc.F.setIdentity();
    sc.F(0, 2) = dt;
    sc.F(1, 3) = dt;

    // Discrete white-noise acceleration model.
    const double q = accel_sigma * accel_sigma;
    const double dt2 = dt * dt, dt3 = dt2 * dt / 2, dt4 = dt2 * dt2 / 4;
    sc.Q.setZero();
    for (int a = 0; a < 2; ++a) {
        sc.Q(a, a) = dt4 * q;
        sc.Q(a, a + 2) = sc.Q(a + 2, a) = dt3 * q;
        sc.Q(a + 2, a + 2) = dt2 * q;
    }

    sc.H.setZero();
    sc.H(0, 0) = sc.H(1, 1) = 1.0;
    sc.R = Eigen::Matrix2d::Identity() * meas_sigma * meas_sigma;

    sc.x0 << 0.0, 0.0, 1.0, 0.5;
    sc.P0 = Eigen::Vector4d(10.0, 10.0, 4.0, 4.0).asDiagonal();

    std::mt19937 rng(seed);
    Eigen::Vector4d x = sc.x0 + sample<4>(rng, sc.P0);
    for (int k = 0; k < steps; ++k) {
        x = sc.F * x + sample<4>(rng, sc.Q);
        sc.truth.push_back(x);
        sc.measurements.push_back(sc.H * x + sample<2>(rng, sc.R));
    }
    return sc;
}

}  // namespace scenarios
