// Tutorial 1: a constant-velocity tracker, smoothed and checked for consistency.
// See docs/tutorials/01-constant-velocity-tracker.md.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <kalman/kalman.hpp>

int main() {
    // Model: state [px, py, vx, vy], a constant-velocity target observed by a
    // position sensor at 10 Hz. Unmodeled accelerations enter as process noise.
    constexpr double dt = 0.1;
    constexpr double accel_sigma = 0.5;  // m/s^2
    constexpr double meas_sigma = 1.0;   // m
    using KF = kalman::LinearFilter<4, 2>;

    KF::Transition F = KF::Transition::Identity();
    F(0, 2) = F(1, 3) = dt;

    KF::Observation H = KF::Observation::Zero();
    H(0, 0) = H(1, 1) = 1.0;

    // Discrete white-noise acceleration: per axis q * [dt^4/4, dt^3/2; dt^3/2, dt^2].
    KF::Cov Q = KF::Cov::Zero();
    const double q = accel_sigma * accel_sigma;
    for (int axis = 0; axis < 2; ++axis) {
        Q(axis, axis) = q * std::pow(dt, 4) / 4;
        Q(axis, axis + 2) = Q(axis + 2, axis) = q * std::pow(dt, 3) / 2;
        Q(axis + 2, axis + 2) = q * dt * dt;
    }
    const KF::MeasCov R = meas_sigma * meas_sigma * KF::MeasCov::Identity();

    // Simulate a target. The filter starts with a rough guess and a wide prior.
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 1.0);
    Eigen::Vector4d truth(0.0, 0.0, 3.0, -1.0);
    const KF::State x0(5.0, -5.0, 0.0, 0.0);
    const KF::Cov P0 = Eigen::Vector4d(100.0, 100.0, 25.0, 25.0).asDiagonal();

    KF kf(F, H, Q, R, x0, P0);
    kalman::RtsSmoother<4> smoother;
    smoother.record(kf);  // the initial state

    std::vector<Eigen::Vector4d> truths, filtered;
    double nis_sum = 0.0;
    const int steps = 300;
    for (int k = 0; k < steps; ++k) {
        const Eigen::Vector2d accel(accel_sigma * noise(rng), accel_sigma * noise(rng));
        truth.head<2>() += truth.tail<2>() * dt + 0.5 * accel * dt * dt;
        truth.tail<2>() += accel * dt;
        const KF::Measurement z = truth.head<2>() + meas_sigma * Eigen::Vector2d(noise(rng), noise(rng));

        kf.predict();
        if (!kf.update(z)) return 1;  // the innovation covariance was not positive-definite
        smoother.record(kf);

        nis_sum += kf.nis();
        truths.push_back(truth);
        filtered.push_back(kf.state());
    }

    // The smoother revisits every step using the measurements that came after it.
    const auto smoothed = smoother.smooth();  // one entry per record; [0] is the initial state
    double filtered_sq = 0.0, smoothed_sq = 0.0;
    for (int k = 0; k < steps; ++k) {
        filtered_sq += (filtered[k] - truths[k]).head<2>().squaredNorm();
        smoothed_sq += (smoothed[k + 1].x - truths[k]).head<2>().squaredNorm();
    }
    const double filtered_rmse = std::sqrt(filtered_sq / steps);
    const double smoothed_rmse = std::sqrt(smoothed_sq / steps);

    // Consistency: for a well-tuned filter the NIS averages to the measurement
    // dimension (2), within a chi-square interval.
    const double mean_nis = nis_sum / steps;
    const auto bounds = kalman::diagnostics::average_bounds(/*dof=*/2, /*samples=*/steps);

    std::printf("position RMSE: measurements %.2f m, filtered %.2f m, smoothed %.2f m\n", meas_sigma, filtered_rmse,
                smoothed_rmse);
    std::printf("mean NIS %.2f (95%% interval for a consistent filter: [%.2f, %.2f])\n", mean_nis, bounds.lower,
                bounds.upper);

    // Checks for the example test: filtering beats raw measurements, smoothing
    // beats filtering, and the NIS is plausible.
    const bool ok = filtered_rmse < meas_sigma && smoothed_rmse < filtered_rmse && mean_nis > 1.0 && mean_nis < 3.0;
    return ok ? 0 : 1;
}
