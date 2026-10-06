#pragma once

#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include "kalman/concepts.hpp"
#include "kalman/detail/unscented.hpp"
#include "kalman/prediction.hpp"

namespace kalman {

// Square-root unscented Kalman filter (Van der Merwe & Wan, 2001).
//
// Propagates the lower Cholesky factor S of the covariance (P = S S^T) through
// QR decompositions and rank-1 Cholesky updates, so P is positive
// semi-definite by construction, and the factor needs no re-decomposition at
// each step. Same model interface as UnscentedKalmanFilter.
template <int N, typename Scalar = double>
class SquareRootUnscentedKalmanFilter {
    static_assert(N > 0, "state size must be positive");
    static constexpr int L = 2 * N + 1;

public:
    using State = Eigen::Matrix<Scalar, N, 1>;
    using Cov = Eigen::Matrix<Scalar, N, N>;

    // Throws std::invalid_argument if P0 is not positive-definite.
    SquareRootUnscentedKalmanFilter(const State& x0, const Cov& P0, const UnscentedParams& params = {})
        : x_(x0), w_(params) {
        if (!set_state(x0, P0)) throw std::invalid_argument("P0 must be positive-definite");
    }

    // Q may be positive semi-definite. Returns false (and leaves the estimate
    // untouched) if the propagated factor would be singular.
    template <ProcessModel<N, Scalar> Model>
    bool predict(const Model& f, double dt, const Cov& Q) {
        const auto X = detail::sigma_points<N, Scalar>(x_, S_, w_.gamma);

        Eigen::Matrix<Scalar, N, L> Y;
        for (int i = 0; i < L; ++i) Y.col(i) = f.predict(State(X.col(i)), dt);

        const State x = Y * w_.wm;
        const Eigen::Matrix<Scalar, N, L> D = Y.colwise() - x;
        Cov S;
        if (!detail::sqrt_covariance(D, w_.wc, detail::psd_sqrt(Q), S)) return false;

        Cov cross = Cov::Zero();
        for (int i = 0; i < L; ++i) cross += w_.wc(i) * (X.col(i) - x_) * D.col(i).transpose();
        x_ = x;
        S_ = S;
        pred_.x = x_;
        pred_.P = S_ * S_.transpose();
        pred_.cross = cross;
        ++pred_.sequence;
        return true;
    }

    // R must be positive-definite. Returns false (and leaves the estimate
    // untouched) if R is not, or if the covariance downdate fails.
    template <int Mz, typename Model>
        requires MeasurementModel<Model, N, Mz, Scalar>
    bool update(const Model& h, const Eigen::Matrix<Scalar, Mz, 1>& z, const Eigen::Matrix<Scalar, Mz, Mz>& R) {
        using Meas = Eigen::Matrix<Scalar, Mz, 1>;
        using MeasCov = Eigen::Matrix<Scalar, Mz, Mz>;

        const Eigen::LLT<MeasCov> r_llt(R);
        if (r_llt.info() != Eigen::Success) return false;

        const auto X = detail::sigma_points<N, Scalar>(x_, S_, w_.gamma);
        Eigen::Matrix<Scalar, Mz, L> Z;
        for (int i = 0; i < L; ++i) Z.col(i) = h.measure(State(X.col(i)));
        const Meas z_pred = detail::sigma_mean(h, Z, w_.wm);

        Eigen::Matrix<Scalar, Mz, L> DZ;
        Eigen::Matrix<Scalar, N, Mz> Pxz = Eigen::Matrix<Scalar, N, Mz>::Zero();
        for (int i = 0; i < L; ++i) {
            DZ.col(i) = detail::measurement_residual(h, Meas(Z.col(i)), z_pred);
            Pxz += w_.wc(i) * (X.col(i) - x_) * DZ.col(i).transpose();
        }

        MeasCov Sz;
        if (!detail::sqrt_covariance(DZ, w_.wc, MeasCov(r_llt.matrixL()), Sz)) return false;

        // K = Pxz (Sz Sz^T)^-1, via two triangular solves.
        const Eigen::Matrix<Scalar, Mz, N> tmp = Sz.template triangularView<Eigen::Lower>().solve(Pxz.transpose());
        const Eigen::Matrix<Scalar, N, Mz> K =
            Sz.transpose().template triangularView<Eigen::Upper>().solve(tmp).transpose();

        const Meas y = detail::measurement_residual(h, z, z_pred);

        // P -= K Pzz K^T = (K Sz)(K Sz)^T, one rank-1 downdate per column.
        Cov S = S_;
        const Eigen::Matrix<Scalar, N, Mz> U = K * Sz;
        for (int j = 0; j < Mz; ++j) {
            if (!detail::cholupdate<N, Scalar>(S, U.col(j), Scalar(-1))) return false;
        }

        x_ += K * y;
        S_ = S;
        nis_ = Sz.template triangularView<Eigen::Lower>().solve(y).squaredNorm();
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

    // Returns false (and leaves the estimate untouched) if P is not
    // positive-definite.
    bool set_state(const State& x, const Cov& P) {
        const Eigen::LLT<Cov> llt(P);
        if (llt.info() != Eigen::Success) return false;
        x_ = x;
        S_ = llt.matrixL();
        return true;
    }

private:
    State x_;
    Cov S_;
    detail::UnscentedWeights<N, Scalar> w_;
    Scalar nis_ = Scalar(0);
    Prediction<N, Scalar> pred_;
};

}  // namespace kalman
