#pragma once

#include <Eigen/Dense>

#include "kalman/detail/small_cholesky.hpp"

namespace kalman::detail {

template <typename Mat>
void symmetrize(Mat& A) {
    A = typename Mat::Scalar(0.5) * (A + A.transpose()).eval();
}

// Measurement update shared by every Gaussian filter with a linearized model:
//
//   S = H P H^T + R,   K = P H^T S^-1
//   x = x + K y,       P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form)
//
// The Joseph form keeps P symmetric positive-definite under round-off. Returns
// false, leaving x and P untouched, if S is not positive-definite.
template <typename Scalar, int N, int M>
bool joseph_update(Eigen::Matrix<Scalar, N, 1>& x, Eigen::Matrix<Scalar, N, N>& P, const Eigen::Matrix<Scalar, M, N>& H,
                   const Eigen::Matrix<Scalar, M, 1>& y, const Eigen::Matrix<Scalar, M, M>& R, Scalar& nis) {
    using Cov = Eigen::Matrix<Scalar, N, N>;

    const Eigen::Matrix<Scalar, M, N> HP = H * P;
    Eigen::Matrix<Scalar, M, M> S = HP * H.transpose() + R;
    symmetrize(S);

    const SmallCholesky<Scalar, M> llt(S);
    if (!llt.ok()) return false;

    // S is symmetric, so K^T = S^-1 (H P).
    const Eigen::Matrix<Scalar, N, M> K = llt.template solve<N>(HP).transpose();
    const Cov I_KH = Cov::Identity() - K * H;

    x += K * y;
    P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
    symmetrize(P);

    nis = y.dot(llt.template solve<1>(y));
    return true;
}

}  // namespace kalman::detail
