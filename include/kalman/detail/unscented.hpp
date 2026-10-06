#pragma once

#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

#include "kalman/concepts.hpp"

namespace kalman {

// Scaled unscented transform parameters (Van der Merwe). The defaults keep all
// weights non-negative, which is the robust choice for strongly nonlinear
// models; alpha = 1e-3 is the other common setting.
struct UnscentedParams {
    double alpha = 1.0;
    double beta = 2.0;
    double kappa = 0.0;
};

namespace detail {

template <int N, typename Scalar>
struct UnscentedWeights {
    static constexpr int L = 2 * N + 1;

    Scalar gamma;  // sigma-point spread: sqrt(N + lambda)
    Eigen::Matrix<Scalar, L, 1> wm, wc;

    explicit UnscentedWeights(const UnscentedParams& p) {
        const double c = p.alpha * p.alpha * (N + p.kappa);  // N + lambda
        if (!(c > 0.0)) throw std::invalid_argument("unscented parameters need alpha^2 (N + kappa) > 0");
        const double lambda = c - N;
        gamma = static_cast<Scalar>(std::sqrt(c));
        wm.setConstant(static_cast<Scalar>(0.5 / c));
        wc = wm;
        wm(0) = static_cast<Scalar>(lambda / c);
        wc(0) = static_cast<Scalar>(lambda / c + 1.0 - p.alpha * p.alpha + p.beta);
    }
};

// Columns are x, x + gamma * S_i, x - gamma * S_i, where S S^T = P.
template <int N, typename Scalar>
Eigen::Matrix<Scalar, N, 2 * N + 1> sigma_points(const Eigen::Matrix<Scalar, N, 1>& x,
                                                 const Eigen::Matrix<Scalar, N, N>& S, Scalar gamma) {
    Eigen::Matrix<Scalar, N, 2 * N + 1> X;
    X.col(0) = x;
    for (int i = 0; i < N; ++i) {
        X.col(1 + i) = x + gamma * S.col(i);
        X.col(1 + N + i) = x - gamma * S.col(i);
    }
    return X;
}

template <typename Model, int Mz, typename Scalar>
Eigen::Matrix<Scalar, Mz, 1> measurement_residual(const Model& h, const Eigen::Matrix<Scalar, Mz, 1>& a,
                                                  const Eigen::Matrix<Scalar, Mz, 1>& b) {
    if constexpr (HasResidual<Model, Mz, Scalar>) {
        return h.residual(a, b);
    } else {
        return a - b;
    }
}

// Weighted mean of sigma points. With a residual() hook, the mean is taken
// around the central point so that wrapped quantities (angles) average
// correctly across the branch cut.
template <typename Model, int Mz, int L, typename Scalar>
Eigen::Matrix<Scalar, Mz, 1> sigma_mean(const Model& h, const Eigen::Matrix<Scalar, Mz, L>& Z,
                                        const Eigen::Matrix<Scalar, L, 1>& wm) {
    if constexpr (HasResidual<Model, Mz, Scalar>) {
        const Eigen::Matrix<Scalar, Mz, 1> ref = Z.col(0);
        Eigen::Matrix<Scalar, Mz, 1> acc = Eigen::Matrix<Scalar, Mz, 1>::Zero();
        for (int i = 0; i < L; ++i) acc += wm(i) * h.residual(Eigen::Matrix<Scalar, Mz, 1>(Z.col(i)), ref);
        return ref + acc;
    } else {
        return Z * wm;
    }
}

// Any A with A A^T = M, for symmetric positive semi-definite M (e.g. a
// rank-deficient process noise). Small negative eigenvalues from round-off are
// clipped to zero.
template <int K, typename Scalar>
Eigen::Matrix<Scalar, K, K> psd_sqrt(const Eigen::Matrix<Scalar, K, K>& M) {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, K, K>> es(M);
    return es.eigenvectors() * es.eigenvalues().cwiseMax(Scalar(0)).cwiseSqrt().asDiagonal();
}

// Rank-1 update of a lower Cholesky factor in place: L L^T += sign * v v^T.
// Returns false if a downdate would make the matrix indefinite.
template <int K, typename Scalar>
bool cholupdate(Eigen::Matrix<Scalar, K, K>& L, Eigen::Matrix<Scalar, K, 1> v, Scalar sign) {
    for (int k = 0; k < K; ++k) {
        const Scalar r2 = L(k, k) * L(k, k) + sign * v(k) * v(k);
        if (!(r2 > Scalar(0))) return false;
        const Scalar r = std::sqrt(r2);
        const Scalar c = r / L(k, k);
        const Scalar s = v(k) / L(k, k);
        L(k, k) = r;
        const int rest = K - k - 1;
        if (rest > 0) {
            L.col(k).tail(rest) = (L.col(k).tail(rest) + sign * s * v.tail(rest)) / c;
            v.tail(rest) = c * v.tail(rest) - s * L.col(k).tail(rest);
        }
    }
    return true;
}

// Lower Cholesky factor of  sum_{i>=1} wc_i d_i d_i^T + B B^T + wc_0 d_0 d_0^T
// via one QR decomposition plus a rank-1 update for the central point, which
// may carry a negative weight. Requires wc_i > 0 for i >= 1.
template <int K, int L, int B, typename Scalar>
bool sqrt_covariance(const Eigen::Matrix<Scalar, K, L>& D, const Eigen::Matrix<Scalar, L, 1>& wc,
                     const Eigen::Matrix<Scalar, K, B>& sqrt_noise, Eigen::Matrix<Scalar, K, K>& S) {
    Eigen::Matrix<Scalar, L - 1 + B, K> A;
    A.topRows(L - 1) = (D.rightCols(L - 1) * std::sqrt(wc(1))).transpose();
    A.bottomRows(B) = sqrt_noise.transpose();

    const Eigen::HouseholderQR<decltype(A)> qr(A);
    S = qr.matrixQR().template topRows<K>().template triangularView<Eigen::Upper>().transpose();
    // QR fixes R only up to the sign of each row; make the diagonal positive.
    for (int k = 0; k < K; ++k) {
        if (S(k, k) < Scalar(0)) S.col(k) = -S.col(k);
    }

    if (wc(0) != Scalar(0)) {
        const Eigen::Matrix<Scalar, K, 1> v = std::sqrt(std::abs(wc(0))) * D.col(0);
        if (!cholupdate(S, v, wc(0) > Scalar(0) ? Scalar(1) : Scalar(-1))) return false;
    }
    return (S.diagonal().array() > Scalar(0)).all();
}

}  // namespace detail
}  // namespace kalman
