#pragma once

#include <Eigen/Dense>

#include "kalman/autodiff.hpp"
#include "kalman/concepts.hpp"
#include "kalman/detail/joseph_update.hpp"
#include "kalman/prediction.hpp"

namespace kalman {

// Extended Kalman filter with fixed state size N.
//
// Models are passed per call, so one filter can fuse several sensors with
// different measurement sizes; each size is still checked at compile time.
//
// Jacobians come from the model when it provides them (`jacobian(x, dt)` for
// process models, `jacobian(x)` for measurement models). Otherwise the model
// must be templated on its scalar type and the Jacobian is computed by
// forward-mode autodiff:
//
//   struct Model {
//       template <typename T>
//       Eigen::Matrix<T, N, 1> predict(const Eigen::Matrix<T, N, 1>& x, double dt) const;
//   };
//
// Measurement models may define `residual(z, z_pred)` for quantities such as
// angles, where the innovation is not a plain subtraction.
template <int N, typename Scalar = double>
class ExtendedKalmanFilter {
    static_assert(N > 0, "state size must be positive");

public:
    using State = Eigen::Matrix<Scalar, N, 1>;
    using Cov = Eigen::Matrix<Scalar, N, N>;

    ExtendedKalmanFilter(const State& x0, const Cov& P0) : x_(x0), P_(P0) {}

    template <ProcessModel<N, Scalar> Model>
    void predict(const Model& f, double dt, const Cov& Q) {
        Cov F;
        if constexpr (AnalyticProcessModel<Model, N, Scalar>) {
            F = f.jacobian(x_, dt);
            x_ = f.predict(x_, dt);
        } else if constexpr (AutodiffProcessModel<Model, N, Scalar>) {
            auto [x_next, J] = value_and_jacobian([&](const auto& x) { return f.predict(x, dt); }, x_);
            F = J;
            x_ = x_next;
        } else {
            static_assert(sizeof(Model) == 0,
                          "process model needs either jacobian(x, dt) or a predict() templated on the scalar");
        }
        pred_.cross = P_ * F.transpose();
        P_ = F * P_ * F.transpose() + Q;
        detail::symmetrize(P_);
        pred_.x = x_;
        pred_.P = P_;
        ++pred_.sequence;
    }

    // Returns false (and leaves the estimate untouched) if the innovation
    // covariance is not positive-definite.
    template <int Mz, typename Model>
        requires MeasurementModel<Model, N, Mz, Scalar>
    bool update(const Model& h, const Eigen::Matrix<Scalar, Mz, 1>& z, const Eigen::Matrix<Scalar, Mz, Mz>& R) {
        Eigen::Matrix<Scalar, Mz, 1> z_pred;
        Eigen::Matrix<Scalar, Mz, N> H;
        if constexpr (AnalyticMeasurementModel<Model, N, Mz, Scalar>) {
            z_pred = h.measure(x_);
            H = h.jacobian(x_);
        } else if constexpr (AutodiffMeasurementModel<Model, N, Mz, Scalar>) {
            auto [value, J] = value_and_jacobian(h, x_);
            z_pred = value;
            H = J;
        } else {
            static_assert(sizeof(Model) == 0,
                          "measurement model needs either jacobian(x) or a measure() templated on the scalar");
        }

        Eigen::Matrix<Scalar, Mz, 1> y;
        if constexpr (HasResidual<Model, Mz, Scalar>) {
            y = h.residual(z, z_pred);
        } else {
            y = z - z_pred;
        }
        return detail::joseph_update(x_, P_, H, y, R, nis_);
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
    Scalar nis_ = Scalar(0);
    Prediction<N, Scalar> pred_;
};

}  // namespace kalman
