#pragma once

#include <random>

#include <Eigen/Dense>

#include "scenarios/constant_velocity.hpp"

namespace scenarios {

struct IllConditionedParams {
    double dt = 1.0;
    double accel_sigma = 1e-4;  // nearly deterministic motion
    double meas_sigma = 1e-3;   // very precise position measurements
    double pos_sigma0 = 1e3;    // ...against a very vague prior
    double vel_sigma0 = 1e2;
};

// A constant-velocity problem whose covariance spans many orders of
// magnitude, which stresses single-precision covariance updates. The truth is
// simulated in double; measure() must be called with k = 0, 1, 2, ...
template <typename Scalar>
class IllConditioned {
public:
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using Mat4 = Eigen::Matrix<Scalar, 4, 4>;

    Mat4 F, Q, P0;
    Eigen::Matrix<Scalar, 2, 4> H;
    Eigen::Matrix<Scalar, 2, 2> R;
    Vec4 x0;

    IllConditioned(unsigned seed, const IllConditionedParams& p)
        : rng_(seed),
          Fd_(cv_transition(p.dt)),
          Qd_(cv_process_noise(p.dt, p.accel_sigma)),
          Rd_(Eigen::Matrix2d::Identity() * p.meas_sigma * p.meas_sigma) {
        Eigen::Matrix<double, 2, 4> Hd = Eigen::Matrix<double, 2, 4>::Zero();
        Hd(0, 0) = Hd(1, 1) = 1.0;
        const double ps = p.pos_sigma0 * p.pos_sigma0, vs = p.vel_sigma0 * p.vel_sigma0;
        const Eigen::Matrix4d P0d = Eigen::Vector4d(ps, ps, vs, vs).asDiagonal();
        const Eigen::Vector4d x0d(0.0, 0.0, 1.0, -1.0);

        F = Fd_.cast<Scalar>();
        Q = Qd_.cast<Scalar>();
        H = Hd.cast<Scalar>();
        R = Rd_.cast<Scalar>();
        P0 = P0d.cast<Scalar>();
        x0 = x0d.cast<Scalar>();
        truth_ = x0d + sample<4>(rng_, P0d);
        Hd_ = Hd;
    }

    Eigen::Matrix<Scalar, 2, 1> measure(long /*k*/) {
        truth_ = Fd_ * truth_ + sample<4>(rng_, Qd_);
        return (Hd_ * truth_ + sample<2>(rng_, Rd_)).template cast<Scalar>();
    }

    const Eigen::Vector4d& truth() const { return truth_; }

private:
    std::mt19937 rng_;
    Eigen::Matrix4d Fd_, Qd_;
    Eigen::Matrix2d Rd_;
    Eigen::Matrix<double, 2, 4> Hd_;
    Eigen::Vector4d truth_;
};

template <typename Scalar>
IllConditioned<Scalar> ill_conditioned(unsigned seed, const IllConditionedParams& p = {}) {
    return IllConditioned<Scalar>(seed, p);
}

}  // namespace scenarios
