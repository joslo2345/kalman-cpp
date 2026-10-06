#pragma once

#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include "kalman/detail/unscented.hpp"
#include "kalman/prediction.hpp"

namespace kalman {

// Square-root linear Kalman filter: propagates a lower-triangular factor S of
// the covariance (P = S S^T) with orthogonal transformations (the "array"
// form of Kailath and Morf), so P is positive semi-definite by construction.
//
// Use it when the covariance spans more orders of magnitude than the scalar
// type can hold, e.g. a vague prior against very precise measurements in
// single precision: the covariance form rounds the small variances away (a
// 1e-6 variance added to 1e4 is lost in float), while the factor keeps a
// 1e-3 standard deviation next to 1e2.
//
//   predict:  [F S, sqrt(Q)]                     --QR-->  [S_pred, 0]
//   update:   [[sqrt(R), H S], [0, S]]           --QR-->  [[Sz, 0], [G, S_post]]
//             K = G Sz^-1,  x = x + K (z - H x)
template <int N, int M, typename Scalar = double>
class SquareRootLinearFilter {
    static_assert(N > 0 && M > 0, "state and measurement sizes must be positive");

public:
    using State = Eigen::Matrix<Scalar, N, 1>;
    using Cov = Eigen::Matrix<Scalar, N, N>;
    using Measurement = Eigen::Matrix<Scalar, M, 1>;
    using MeasCov = Eigen::Matrix<Scalar, M, M>;
    using Transition = Eigen::Matrix<Scalar, N, N>;
    using Observation = Eigen::Matrix<Scalar, M, N>;

    // Throws std::invalid_argument if P0 or R is not positive-definite. Q may
    // be positive semi-definite.
    SquareRootLinearFilter(const Transition& F, const Observation& H, const Cov& Q, const MeasCov& R, const State& x0,
                           const Cov& P0)
        : F_(F), H_(H), sqrt_Q_(detail::psd_sqrt(Q)), x_(x0) {
        const Eigen::LLT<MeasCov> r_llt(R);
        const Eigen::LLT<Cov> p_llt(P0);
        if (r_llt.info() != Eigen::Success) throw std::invalid_argument("R must be positive-definite");
        if (p_llt.info() != Eigen::Success) throw std::invalid_argument("P0 must be positive-definite");
        sqrt_R_ = r_llt.matrixL();
        S_ = p_llt.matrixL();
    }

    void predict() {
        Eigen::Matrix<Scalar, 2 * N, N> A;
        A.template topRows<N>() = (F_ * S_).transpose();
        A.template bottomRows<N>() = sqrt_Q_.transpose();

        pred_.cross = S_ * (F_ * S_).transpose();  // P F^T = S (F S)^T
        x_ = F_ * x_;
        S_ = lower_factor<N>(A);
        pred_.x = x_;
        pred_.P = S_ * S_.transpose();
        ++pred_.sequence;
    }

    // Returns false (and leaves the estimate untouched) if the innovation
    // covariance is singular to working precision.
    bool update(const Measurement& z) {
        // Rows of the transposed pre-array [[sqrt(R), H S], [0, S]]^T.
        Eigen::Matrix<Scalar, M + N, M + N> A = Eigen::Matrix<Scalar, M + N, M + N>::Zero();
        A.template topLeftCorner<M, M>() = sqrt_R_.transpose();
        A.template bottomLeftCorner<N, M>() = (H_ * S_).transpose();
        A.template bottomRightCorner<N, N>() = S_.transpose();
        const Eigen::Matrix<Scalar, M + N, M + N> post = lower_factor<M + N>(A);

        const MeasCov Sz = post.template topLeftCorner<M, M>();
        if (!(Sz.diagonal().array() > Scalar(0)).all()) return false;
        const Eigen::Matrix<Scalar, N, M> G = post.template bottomLeftCorner<N, M>();

        const Measurement y = z - H_ * x_;
        const Measurement w = Sz.template triangularView<Eigen::Lower>().solve(y);  // Sz^-1 y
        // K y = G Sz^-1 y.
        x_ += G * w;
        S_ = post.template bottomRightCorner<N, N>();
        nis_ = w.squaredNorm();
        return true;
    }

    const State& state() const { return x_; }
    Cov covariance() const { return S_ * S_.transpose(); }
    // Lower-triangular factor S with P = S S^T.
    const Cov& sqrt_covariance() const { return S_; }

    // Normalized innovation squared from the last successful update.
    Scalar nis() const { return nis_; }

    // Mean, covariance and cross-covariance from the last predict(), for smoothing.
    const Prediction<N, Scalar>& last_prediction() const { return pred_; }

private:
    // Lower-triangular L with L L^T = A^T A, from the QR decomposition of A
    // (rows >= K), with a non-negative diagonal.
    template <int K, int Rows>
    static Eigen::Matrix<Scalar, K, K> lower_factor(const Eigen::Matrix<Scalar, Rows, K>& A) {
        const Eigen::HouseholderQR<Eigen::Matrix<Scalar, Rows, K>> qr(A);
        Eigen::Matrix<Scalar, K, K> L =
            qr.matrixQR().template topRows<K>().template triangularView<Eigen::Upper>().transpose();
        for (int k = 0; k < K; ++k) {
            if (L(k, k) < Scalar(0)) L.col(k) = -L.col(k);
        }
        return L;
    }

    Transition F_;
    Observation H_;
    Cov sqrt_Q_;
    MeasCov sqrt_R_;
    State x_;
    Cov S_;
    Scalar nis_ = Scalar(0);
    Prediction<N, Scalar> pred_;
};

}  // namespace kalman
