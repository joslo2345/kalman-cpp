#pragma once

#include <cstdint>

#include <Eigen/Dense>

namespace kalman {

// What a filter's predict() step produced, kept so that a smoother can run the
// backward pass later.
template <int N, typename Scalar = double>
struct Prediction {
    Eigen::Matrix<Scalar, N, 1> x = Eigen::Matrix<Scalar, N, 1>::Zero();      // x_{k|k-1}
    Eigen::Matrix<Scalar, N, N> P = Eigen::Matrix<Scalar, N, N>::Zero();      // P_{k|k-1}
    Eigen::Matrix<Scalar, N, N> cross = Eigen::Matrix<Scalar, N, N>::Zero();  // Cov(x_{k-1}, x_k | z_{1:k-1})
    std::uint64_t sequence = 0;  // incremented by every predict(); 0 means none yet
};

}  // namespace kalman
