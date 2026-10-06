#pragma once

#include <Eigen/Dense>

#include "kalman/detail/joseph_update.hpp"
#include "kalman/prediction.hpp"

namespace kalman {

// Linear Kalman filter with fixed state size N and measurement size M.
//
//   predict:  x = F x,   P = F P F^T + Q
//   update:   y = z - H x,   S = H P H^T + R,   K = P H^T S^-1
//             x = x + K y,   P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form)
//
// The Joseph form keeps P symmetric positive-definite under round-off, which is
// what lets the filter run for long horizons in single precision.
template <int N, int M, typename Scalar = double>
class LinearFilter {
    static_assert(N > 0 && M > 0, "state and measurement sizes must be positive");

public:
    using State = Eigen::Matrix<Scalar, N, 1>;
    using Cov = Eigen::Matrix<Scalar, N, N>;
    using Measurement = Eigen::Matrix<Scalar, M, 1>;
    using MeasCov = Eigen::Matrix<Scalar, M, M>;
    using Transition = Eigen::Matrix<Scalar, N, N>;
    using Observation = Eigen::Matrix<Scalar, M, N>;
    using Gain = Eigen::Matrix<Scalar, N, M>;

    LinearFilter(const Transition& F, const Observation& H, const Cov& Q, const MeasCov& R, const State& x0,
                 const Cov& P0)
        : F_(F), H_(H), Q_(Q), R_(R), x_(x0), P_(P0) {}

    void predict() {
        pred_.cross = P_ * F_.transpose();
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
        detail::symmetrize(P_);
        record_prediction();
    }

    // Returns false (and leaves the estimate untouched) if the innovation
    // covariance is not positive-definite.
    bool update(const Measurement& z) { return update(z, R_); }

    bool update(const Measurement& z, const MeasCov& R) {
        const Measurement y = z - H_ * x_;
        if (!detail::joseph_update(x_, P_, H_, y, R, nis_)) return false;
        innovation_ = y;
        return true;
    }

    const State& state() const { return x_; }
    const Cov& covariance() const { return P_; }

    // Normalized innovation squared from the last successful update.
    Scalar nis() const { return nis_; }
    const Measurement& innovation() const { return innovation_; }

    // Mean, covariance and cross-covariance from the last predict(), for smoothing.
    const Prediction<N, Scalar>& last_prediction() const { return pred_; }

    void set_state(const State& x, const Cov& P) {
        x_ = x;
        P_ = P;
    }
    void set_transition(const Transition& F) { F_ = F; }
    void set_observation(const Observation& H) { H_ = H; }
    void set_process_noise(const Cov& Q) { Q_ = Q; }
    void set_measurement_noise(const MeasCov& R) { R_ = R; }

    const Transition& transition() const { return F_; }
    const Observation& observation() const { return H_; }
    const Cov& process_noise() const { return Q_; }
    const MeasCov& measurement_noise() const { return R_; }

private:
    void record_prediction() {
        pred_.x = x_;
        pred_.P = P_;
        ++pred_.sequence;
    }

    Transition F_;
    Observation H_;
    Cov Q_;
    MeasCov R_;
    State x_;
    Cov P_;
    Measurement innovation_ = Measurement::Zero();
    Scalar nis_ = Scalar(0);
    Prediction<N, Scalar> pred_;
};

}  // namespace kalman
