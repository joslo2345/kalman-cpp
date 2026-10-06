// Tutorial 3: attitude estimation with an error-state filter on SO(3).
// See docs/tutorials/03-attitude-estimation.md.
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>

#include <kalman/kalman.hpp>

namespace {

// Orientation plus gyro bias: a 6-DoF product of SO(3) and R^3. Templated on
// the scalar so the filter can differentiate through [+] and [-].
template <typename T>
struct AttitudeBias {
    static constexpr int DoF = 6;

    kalman::SO3<T> R;
    Eigen::Matrix<T, 3, 1> bias = Eigen::Matrix<T, 3, 1>::Zero();

    AttitudeBias boxplus(const Eigen::Matrix<T, 6, 1>& d) const {
        return {R.boxplus(d.template head<3>()), bias + d.template tail<3>()};
    }
    Eigen::Matrix<T, 6, 1> boxminus(const AttitudeBias& other) const {
        Eigen::Matrix<T, 6, 1> out;
        out << R.boxminus(other.R), bias - other.bias;
        return out;
    }
    template <typename U>
    AttitudeBias<U> cast() const {
        return {R.template cast<U>(), bias.template cast<U>()};
    }
};

// Process model: integrate the bias-corrected gyro reading.
struct Gyro {
    Eigen::Vector3d omega;
    template <typename T>
    AttitudeBias<T> predict(const AttitudeBias<T>& x, double dt) const {
        const Eigen::Matrix<T, 3, 1> w = (omega.template cast<T>() - x.bias) * dt;
        return {{(x.R.q * kalman::so3::exp(w)).normalized()}, x.bias};
    }
};

// A known world direction seen in the body frame: gravity (accelerometer) or
// the magnetic field (magnetometer).
struct Direction {
    Eigen::Vector3d world;
    template <typename T>
    Eigen::Matrix<T, 3, 1> measure(const AttitudeBias<T>& x) const {
        return x.R.q.conjugate() * world.template cast<T>();
    }
};

double degrees(double rad) { return rad * 180.0 / std::numbers::pi; }

}  // namespace

int main() {
    constexpr double dt = 0.01, gyro_sigma = 0.01, accel_sigma = 0.02, mag_sigma = 0.03;
    const Eigen::Vector3d up(0, 0, 1), field(0.6, 0, 0.8);
    std::mt19937 rng(3);
    std::normal_distribution<double> n(0.0, 1.0);
    auto noise3 = [&](double s) { return Eigen::Vector3d(s * n(rng), s * n(rng), s * n(rng)); };

    // Truth: about 30 degrees away from where the filter starts, with a gyro bias.
    AttitudeBias<double> truth;
    truth.R.q = kalman::so3::exp(Eigen::Vector3d(0.3, -0.2, 0.4));
    truth.bias = Eigen::Vector3d(0.02, -0.01, 0.015);

    using ESKF = kalman::ErrorStateKalmanFilter<AttitudeBias>;
    ESKF::Cov P0 = ESKF::Cov::Zero();
    P0.topLeftCorner<3, 3>() = 0.25 * Eigen::Matrix3d::Identity();        // 0.5 rad
    P0.bottomRightCorner<3, 3>() = 0.0025 * Eigen::Matrix3d::Identity();  // 0.05 rad/s
    ESKF eskf(AttitudeBias<double>{}, P0);

    ESKF::Cov Q = ESKF::Cov::Zero();
    Q.topLeftCorner<3, 3>() = std::pow(gyro_sigma * dt, 2) * Eigen::Matrix3d::Identity();
    Q.bottomRightCorner<3, 3>() = 1e-12 * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Ra = accel_sigma * accel_sigma * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Rm = mag_sigma * mag_sigma * Eigen::Matrix3d::Identity();

    const int steps = 6000;  // 60 s
    for (int k = 1; k <= steps; ++k) {
        const double t = k * dt;
        const Eigen::Vector3d rate(0.3 * std::sin(0.5 * t), 0.2 * std::cos(0.3 * t), 0.1);
        truth.R.q = (truth.R.q * kalman::so3::exp(Eigen::Vector3d(rate * dt))).normalized();

        eskf.predict(Gyro{rate + truth.bias + noise3(gyro_sigma)}, dt, Q);
        if (k % 10 == 0) {  // accelerometer and magnetometer at 10 Hz
            const Eigen::Vector3d accel = truth.R.q.conjugate() * up + noise3(accel_sigma);
            const Eigen::Vector3d mag = truth.R.q.conjugate() * field + noise3(mag_sigma);
            if (!eskf.update(Direction{up}, accel, Ra) || !eskf.update(Direction{field}, mag, Rm)) return 1;
        }
        if (k % 1000 == 0) {
            std::printf("t = %2.0f s   attitude error %6.2f deg   bias error %.4f rad/s\n", t,
                        degrees(truth.R.boxminus(eskf.state().R).norm()), (truth.bias - eskf.state().bias).norm());
        }
    }
    const double final_error = degrees(truth.R.boxminus(eskf.state().R).norm());
    return final_error < 2.0 ? 0 : 1;
}
