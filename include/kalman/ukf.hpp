#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include "kalman/concepts.hpp"
#include "kalman/detail/joseph_update.hpp"
#include "kalman/detail/unscented.hpp"
#include "kalman/prediction.hpp"

namespace kalman {

// Unscented Kalman filter with fixed state size N and additive noise.
//
// Uses the same model interface as ExtendedKalmanFilter, but needs no
// Jacobians: `predict(x, dt)` and `measure(x)` only have to accept plain
// vectors. A measurement model's optional `residual(z, z_pred)` is also used
// when averaging sigma points, so angles near +/-pi are handled correctly.
template <int N, typename Scalar = double>
class UnscentedKalmanFilter {
    static_assert(N > 0, "state size must be positive");
    static constexpr int L = 2 * N + 1;

public:
    using State = Eigen::Matrix<Scalar, N, 1>;
    using Cov = Eigen::Matrix<Scalar, N, N>;

    UnscentedKalmanFilter(const State& x0, const Cov& P0, const UnscentedParams& params = {})
        : x_(x0), P_(P0), w_(params) {}

    // Returns false (and leaves the estimate untouched) if P is not
    // positive-definite.
    template <ProcessModel<N, Scalar> Model>
    bool predict(const Model& f, double dt, const Cov& Q) {
        const Eigen::LLT<Cov> llt(P_);
        if (llt.info() != Eigen::Success) return false;
        const auto X = detail::sigma_points<N, Scalar>(x_, llt.matrixL(), w_.gamma);

        Eigen::Matrix<Scalar, N, L> Y;
        for (int i = 0; i < L; ++i) Y.col(i) = f.predict(State(X.col(i)), dt);

        const State x = Y * w_.wm;
        Cov P = Q;
        Cov cross = Cov::Zero();
        for (int i = 0; i < L; ++i) {
            const State d = Y.col(i) - x;
            P += w_.wc(i) * d * d.transpose();
            cross += w_.wc(i) * (X.col(i) - x_) * d.transpose();
        }
        detail::symmetrize(P);
        x_ = x;
        P_ = P;
        pred_.x = x_;
        pred_.P = P_;
        pred_.cross = cross;
        ++pred_.sequence;
        return true;
    }

    // Returns false (and leaves the estimate untouched) if P or the innovation
    // covariance is not positive-definite.
    template <int Mz, typename Model>
        requires MeasurementModel<Model, N, Mz, Scalar>
    bool update(const Model& h, const Eigen::Matrix<Scalar, Mz, 1>& z,
                const Eigen::Matrix<Scalar, Mz, Mz>& R) {
        using Meas = Eigen::Matrix<Scalar, Mz, 1>;

        const Eigen::LLT<Cov> llt(P_);
        if (llt.info() != Eigen::Success) return false;
        const auto X = detail::sigma_points<N, Scalar>(x_, llt.matrixL(), w_.gamma);

        Eigen::Matrix<Scalar, Mz, L> Z;
        for (int i = 0; i < L; ++i) Z.col(i) = h.measure(State(X.col(i)));
        const Meas z_pred = detail::sigma_mean(h, Z, w_.wm);

        Eigen::Matrix<Scalar, Mz, Mz> S = R;
        Eigen::Matrix<Scalar, N, Mz> Pxz = Eigen::Matrix<Scalar, N, Mz>::Zero();
        for (int i = 0; i < L; ++i) {
            const Meas dz = detail::measurement_residual(h, Meas(Z.col(i)), z_pred);
            const State dx = X.col(i) - x_;
            S += w_.wc(i) * dz * dz.transpose();
            Pxz += w_.wc(i) * dx * dz.transpose();
        }
        detail::symmetrize(S);

        const Eigen::LLT<Eigen::Matrix<Scalar, Mz, Mz>> s_llt(S);
        if (s_llt.info() != Eigen::Success) return false;

        // S is symmetric, so K^T = S^-1 Pxz^T.
        const Eigen::Matrix<Scalar, N, Mz> K = s_llt.solve(Pxz.transpose()).transpose();
        const Meas y = detail::measurement_residual(h, z, z_pred);

        x_ += K * y;
        P_ -= K * S * K.transpose();
        detail::symmetrize(P_);
        nis_ = y.dot(s_llt.solve(y));
        return true;
    }

    const State& state() const { return x_; }
    const Cov& covariance() const { return P_; }

    // Normalized innovation squared from the last successful update.
    Scalar nis() const { return nis_; }

    // Mean, covariance and cross-covariance from the last predict(), for smoothing.
    const Prediction<N, Scalar>& last_prediction() const { return pred_; }

    void set_state(const State& x, const Cov& P) {
        x_ = x;
        P_ = P;
    }

private:
    State x_;
    Cov P_;
    detail::UnscentedWeights<N, Scalar> w_;
    Scalar nis_ = Scalar(0);
    Prediction<N, Scalar> pred_;
};

}  // namespace kalman
