// Tutorial 2: fusing a 100 Hz accelerometer with a 1 Hz GPS that arrives late.
// See docs/tutorials/02-gps-imu-fusion.md.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <kalman/kalman.hpp>

namespace {

// State [px, py, vx, vy, ax, ay]: constant acceleration, driven by white jerk.
struct ConstantAcceleration {
    template <typename T>
    Eigen::Matrix<T, 6, 1> predict(const Eigen::Matrix<T, 6, 1>& x, double dt) const {
        Eigen::Matrix<T, 6, 1> out = x;
        for (int i = 0; i < 2; ++i) {
            out(i) += x(i + 2) * dt + x(i + 4) * (0.5 * dt * dt);
            out(i + 2) += x(i + 4) * dt;
        }
        return out;
    }
};

Eigen::Matrix<double, 6, 6> jerk_noise(double dt, double jerk_sigma) {
    // Per axis q * [dt^5/20, dt^4/8, dt^3/6; dt^4/8, dt^3/3, dt^2/2; dt^3/6, dt^2/2, dt].
    const double q = jerk_sigma * jerk_sigma;
    const double c[3][3] = {{std::pow(dt, 5) / 20, std::pow(dt, 4) / 8, std::pow(dt, 3) / 6},
                            {std::pow(dt, 4) / 8, std::pow(dt, 3) / 3, dt * dt / 2},
                            {std::pow(dt, 3) / 6, dt * dt / 2, dt}};
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    for (int axis = 0; axis < 2; ++axis)
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) Q(axis + 2 * r, axis + 2 * s) = q * c[r][s];
    return Q;
}

// GPS measures position; the accelerometer measures acceleration (already
// rotated into the world frame and gravity-compensated, to keep this 2-D).
struct GpsPosition {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 6, 1>& x) const {
        return x.template head<2>();
    }
};
struct Accelerometer {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 6, 1>& x) const {
        return x.template tail<2>();
    }
};

// Truth: a vehicle driving a circle of radius 50 m at 5 m/s.
constexpr double radius = 50.0, omega = 0.1;
Eigen::Vector2d position(double t) { return radius * Eigen::Vector2d(std::cos(omega * t), std::sin(omega * t)); }
Eigen::Vector2d acceleration(double t) { return -omega * omega * position(t); }

struct Reading {
    double stamp;    // when it was measured
    double arrival;  // when it reaches the fusion engine
    bool gps;
    Eigen::Vector2d z;
};

}  // namespace

int main() {
    constexpr double gps_sigma = 2.0, gps_latency = 0.3, accel_sigma = 0.05, duration = 60.0;
    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 1.0);
    auto noise2 = [&](double s) { return Eigen::Vector2d(s * noise(rng), s * noise(rng)); };

    // Sensor readings, then sorted by when they arrive, as a real system sees them.
    std::vector<Reading> readings;
    for (int i = 1; i <= static_cast<int>(duration * 100); ++i) {
        const double t = i * 0.01;
        readings.push_back({t, t, false, acceleration(t) + noise2(accel_sigma)});
    }
    for (int i = 1; i <= static_cast<int>(duration); ++i) {
        const double t = i * 1.0;
        readings.push_back({t, t + gps_latency, true, position(t) + noise2(gps_sigma)});
    }
    // std::sort with a full tie-break (arrival, then measurement time, then
    // sensor) gives the same order on every standard library. (std::stable_sort
    // would also work, but MSVC's temporary buffer rejects the 16-byte-aligned
    // Eigen vector inside Reading.)
    std::sort(readings.begin(), readings.end(), [](const Reading& a, const Reading& b) {
        if (a.arrival != b.arrival) return a.arrival < b.arrival;
        if (a.stamp != b.stamp) return a.stamp < b.stamp;
        return a.gps < b.gps;
    });

    // Start near the first GPS fix with a wide prior.
    Eigen::Matrix<double, 6, 1> x0 = Eigen::Matrix<double, 6, 1>::Zero();
    x0.head<2>() = position(0.0) + Eigen::Vector2d(3.0, -3.0);
    Eigen::Matrix<double, 6, 1> p0;
    p0 << 25, 25, 25, 25, 1, 1;
    kalman::ExtendedKalmanFilter<6> ekf(x0, p0.asDiagonal());

    // The engine keeps 1 s of history, enough for the 0.3 s GPS latency.
    kalman::AsyncFusion<kalman::ExtendedKalmanFilter<6>, ConstantAcceleration> fusion(
        ekf, /*t0=*/0.0, ConstantAcceleration{}, [](double dt) { return jerk_noise(dt, 0.5); }, /*horizon=*/1.0);

    const Eigen::Matrix2d R_gps = gps_sigma * gps_sigma * Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d R_accel = accel_sigma * accel_sigma * Eigen::Matrix2d::Identity();

    int reordered = 0;
    double sq = 0.0;
    int samples = 0;
    for (const auto& r : readings) {
        const auto result =
            r.gps ? fusion.add(r.stamp, GpsPosition{}, r.z, R_gps) : fusion.add(r.stamp, Accelerometer{}, r.z, R_accel);
        if (result == kalman::FusionResult::too_old || result == kalman::FusionResult::update_failed) return 1;
        reordered += result == kalman::FusionResult::reordered;
        // Score the estimate after the first 10 s, once the filter has converged.
        if (fusion.time() > 10.0) {
            sq += (fusion.state().head<2>() - position(fusion.time())).squaredNorm();
            ++samples;
        }
    }
    const double rmse = std::sqrt(sq / samples);

    std::printf("%d of %zu readings arrived out of order and were inserted at their timestamp\n", reordered,
                readings.size());
    std::printf("position RMSE: GPS alone %.2f m, fused %.2f m\n", gps_sigma * std::sqrt(2.0), rmse);
    return rmse < gps_sigma ? 0 : 1;
}
