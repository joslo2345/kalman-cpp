#pragma once

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace kalman {
namespace so3 {

// All functions are templated on the scalar and call math functions
// unqualified, so they work with autodiff jets. Near zero they switch to Taylor
// expansions, which keeps derivatives exact at the expansion point (where the
// error-state filter differentiates them).

template <typename T>
Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> S;
    S << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
    return S;
}

/// Rotation vector (axis * angle) to unit quaternion.
template <typename T>
Eigen::Quaternion<T> exp(const Eigen::Matrix<T, 3, 1>& w) {
    using std::cos;
    using std::sin;
    using std::sqrt;
    const T theta2 = w.squaredNorm();
    if (theta2 < T(1e-10)) {
        // cos(t/2) ~ 1 - t^2/8,  sin(t/2)/t ~ 1/2 - t^2/48
        const Eigen::Matrix<T, 3, 1> v = w * (T(0.5) - theta2 / T(48));
        return Eigen::Quaternion<T>(T(1) - theta2 / T(8), v(0), v(1), v(2));
    }
    const T theta = sqrt(theta2);
    const Eigen::Matrix<T, 3, 1> v = w * (sin(theta / T(2)) / theta);
    return Eigen::Quaternion<T>(cos(theta / T(2)), v(0), v(1), v(2));
}

/// Unit quaternion to rotation vector, with angle in [0, pi].
template <typename T>
Eigen::Matrix<T, 3, 1> log(const Eigen::Quaternion<T>& q_in) {
    using std::atan2;
    using std::sqrt;
    // q and -q are the same rotation; pick the one with w >= 0 (shortest path).
    const Eigen::Quaternion<T> q = q_in.w() < T(0) ? Eigen::Quaternion<T>(-q_in.coeffs()) : q_in;
    const T n2 = q.vec().squaredNorm();
    if (n2 < T(1e-10)) {
        // 2 atan(n / w) / n ~ (2 / w) (1 - n^2 / (3 w^2))
        const T& w = q.w();
        return q.vec() * (T(2) / w * (T(1) - n2 / (T(3) * w * w)));
    }
    const T n = sqrt(n2);
    return q.vec() * (T(2) * atan2(n, q.w()) / n);
}

}  // namespace so3

/// Orientation as a manifold state for ErrorStateKalmanFilter, with right
/// (body-frame) perturbations: q [+] d = q * exp(d), and a [-] b = log(b^-1 a).
template <typename T>
struct SO3 {
    static constexpr int DoF = 3;
    using Scalar = T;
    using Tangent = Eigen::Matrix<T, 3, 1>;

    Eigen::Quaternion<T> q = Eigen::Quaternion<T>::Identity();

    SO3 boxplus(const Tangent& d) const { return {(q * so3::exp(d)).normalized()}; }
    Tangent boxminus(const SO3& other) const { return so3::log(other.q.conjugate() * q); }

    template <typename U>
    SO3<U> cast() const {
        return {q.template cast<U>()};
    }

    Eigen::Matrix<T, 3, 3> matrix() const { return q.toRotationMatrix(); }
};

}  // namespace kalman
