#include <cmath>
#include <numbers>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/attitude.hpp"

using scenarios::AttitudeBias;
using ESKF = kalman::ErrorStateKalmanFilter<AttitudeBias>;

namespace {

Eigen::Vector3d random_rotation_vector(std::mt19937& rng, double max_angle) {
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, max_angle);
    return Eigen::Vector3d(n(rng), n(rng), n(rng)).normalized() * u(rng);
}

// SO(3) right Jacobian: Exp(w + d) ~ Exp(w) Exp(J_r(w) d).
Eigen::Matrix3d right_jacobian(const Eigen::Vector3d& w) {
    const double t = w.norm();
    const Eigen::Matrix3d W = kalman::so3::skew(w);
    return Eigen::Matrix3d::Identity() - (1 - std::cos(t)) / (t * t) * W + (t - std::sin(t)) / (t * t * t) * W * W;
}

ESKF::Cov initial_covariance() {
    ESKF::Cov P0 = ESKF::Cov::Zero();
    P0.topLeftCorner<3, 3>() = 0.25 * Eigen::Matrix3d::Identity();        // 0.5 rad
    P0.bottomRightCorner<3, 3>() = 0.0025 * Eigen::Matrix3d::Identity();  // 0.05 rad/s
    return P0;
}

ESKF::Cov process_noise(const scenarios::AttitudeScenario& sc) {
    ESKF::Cov Q = ESKF::Cov::Zero();
    Q.topLeftCorner<3, 3>() = std::pow(sc.gyro_sigma * sc.dt, 2) * Eigen::Matrix3d::Identity();
    Q.bottomRightCorner<3, 3>() = 1e-12 * Eigen::Matrix3d::Identity();
    return Q;
}

struct RunResult {
    double final_attitude_error;  // rad
    double final_bias_error;      // rad/s
    double max_norm_drift;        // | |q| - 1 |
    double nees_sum;              // over the second half of the run
    int nees_count;
};

RunResult run(const scenarios::AttitudeScenario& sc) {
    ESKF eskf(AttitudeBias<double>{}, initial_covariance());  // starts at identity, zero bias
    const ESKF::Cov Q = process_noise(sc);
    const Eigen::Matrix3d Ra = sc.accel_sigma * sc.accel_sigma * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Rm = sc.mag_sigma * sc.mag_sigma * Eigen::Matrix3d::Identity();

    RunResult r{0, 0, 0, 0, 0};
    const std::size_t n = sc.truth.size();
    for (std::size_t k = 0; k < n; ++k) {
        eskf.predict(scenarios::GyroModel{sc.gyro[k]}, sc.dt, Q);
        if (k % sc.vector_every == sc.vector_every - 1) {
            REQUIRE(eskf.update(scenarios::DirectionModel{sc.gravity_ref}, sc.accel[k], Ra));
            REQUIRE(eskf.update(scenarios::DirectionModel{sc.mag_ref}, sc.mag[k], Rm));
        }
        r.max_norm_drift = std::max(r.max_norm_drift, std::abs(eskf.state().R.q.norm() - 1.0));
        if (k >= n / 2) {
            // true = estimate [+] e
            const Eigen::Matrix<double, 6, 1> e = sc.truth[k].boxminus(eskf.state());
            r.nees_sum += e.dot(eskf.covariance().ldlt().solve(e));
            ++r.nees_count;
        }
    }
    r.final_attitude_error = sc.truth.back().R.boxminus(eskf.state().R).norm();
    r.final_bias_error = (sc.truth.back().bias - eskf.state().bias).norm();
    return r;
}

}  // namespace

TEST_CASE("SO(3) exp and log are inverse and match Eigen's AngleAxis", "[unit][so3]") {
    std::mt19937 rng(1);
    for (int i = 0; i < 200; ++i) {
        const Eigen::Vector3d w = random_rotation_vector(rng, 0.999 * std::numbers::pi);
        const Eigen::Quaterniond q = kalman::so3::exp(w);
        REQUIRE(std::abs(q.norm() - 1.0) < 1e-15);
        REQUIRE(q.angularDistance(Eigen::Quaterniond(Eigen::AngleAxisd(w.norm(), w.normalized()))) < 1e-14);
        REQUIRE((kalman::so3::log(q) - w).norm() < 1e-12);

        // [+] and [-] are inverse: (q [+] d) [-] q = d.
        const kalman::SO3<double> x{q};
        const Eigen::Vector3d d = random_rotation_vector(rng, 1.0);
        REQUIRE((x.boxplus(d).boxminus(x) - d).norm() < 1e-12);
    }
}

TEST_CASE("SO(3) log takes the shortest path", "[unit][so3]") {
    const Eigen::Quaterniond q = kalman::so3::exp(Eigen::Vector3d(0.0, 0.0, 0.5));
    const Eigen::Quaterniond neg(-q.coeffs());  // same rotation, w < 0
    REQUIRE((kalman::so3::log(neg) - Eigen::Vector3d(0.0, 0.0, 0.5)).norm() < 1e-14);
}

TEST_CASE("SO(3) exp and log stay smooth across the small-angle branch", "[unit][so3][autodiff]") {
    using J = kalman::Jet<double, 3>;
    // The closed form switches to a Taylor expansion at |w| = 1e-5.
    const Eigen::Vector3d dir = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
    for (double angle : {0.0, 1e-9, 0.99e-5, 1.01e-5, 1e-3}) {
        CAPTURE(angle);
        const Eigen::Vector3d w = angle * dir;
        Eigen::Matrix<J, 3, 1> wj;
        for (int i = 0; i < 3; ++i) wj(i) = J(w(i), i);

        // Value: matches AngleAxis (or identity at 0).
        const Eigen::Quaternion<J> qj = kalman::so3::exp(wj);
        const Eigen::Quaterniond q(qj.w().a, qj.x().a, qj.y().a, qj.z().a);
        const Eigen::Quaterniond ref =
            angle == 0.0 ? Eigen::Quaterniond::Identity() : Eigen::Quaterniond(Eigen::AngleAxisd(angle, dir));
        REQUIRE(q.angularDistance(ref) < 1e-15);

        // Derivative of the vector part: (1/2) I to first order.
        Eigen::Matrix3d D;
        D.row(0) = qj.x().v.transpose();
        D.row(1) = qj.y().v.transpose();
        D.row(2) = qj.z().v.transpose();
        REQUIRE((D - 0.5 * Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() < 1e-6);

        // log(exp(w)) = w, with the identity as its Jacobian.
        const Eigen::Matrix<J, 3, 1> back = kalman::so3::log(qj);
        for (int r = 0; r < 3; ++r) {
            REQUIRE(std::abs(back(r).a - w(r)) < 1e-15);
            REQUIRE((back(r).v - Eigen::Vector3d::Unit(r)).cwiseAbs().maxCoeff() < 1e-9);
        }
    }
}

TEST_CASE("Autodiff error-state Jacobian matches the analytic gyro model", "[unit][eskf][autodiff]") {
    std::mt19937 rng(2);
    for (int i = 0; i < 50; ++i) {
        AttitudeBias<double> x;
        x.R.q = kalman::so3::exp(random_rotation_vector(rng, 3.0));
        x.bias = Eigen::Vector3d(0.01, -0.02, 0.03);
        const Eigen::Vector3d omega = random_rotation_vector(rng, 2.0);
        const double dt = 0.05;

        // Random SPD covariance; predicting with Q = 0 gives F P F^T.
        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Random();
        const ESKF::Cov P = A * A.transpose() + 0.1 * ESKF::Cov::Identity();
        ESKF eskf(x, P);
        eskf.predict(scenarios::GyroModel{omega}, dt, ESKF::Cov::Zero());

        const Eigen::Vector3d w = (omega - x.bias) * dt;
        ESKF::Cov F = ESKF::Cov::Identity();
        F.topLeftCorner<3, 3>() = kalman::so3::exp(w).toRotationMatrix().transpose();
        F.topRightCorner<3, 3>() = -right_jacobian(w) * dt;
        REQUIRE((eskf.covariance() - F * P * F.transpose()).cwiseAbs().maxCoeff() < 1e-12);
    }
}

TEST_CASE("Error-state filter recovers attitude and gyro bias from a 30 degree error", "[unit][eskf]") {
    for (unsigned seed = 0; seed < 5; ++seed) {
        CAPTURE(seed);
        const auto r = run(scenarios::attitude(/*duration=*/60.0, seed));
        UNSCOPED_INFO("[report] seed " << seed << ": attitude error " << r.final_attitude_error * 180 / std::numbers::pi
                                       << " deg, bias error " << r.final_bias_error << " rad/s");
        REQUIRE(r.final_attitude_error < 1.0 * std::numbers::pi / 180);
        REQUIRE(r.final_bias_error < 0.005);
        REQUIRE(r.max_norm_drift < 1e-12);
    }
}

TEST_CASE("Error-state filter is consistent (NEES)", "[unit][eskf]") {
    double nees_sum = 0.0;
    int count = 0;
    for (unsigned seed = 100; seed < 120; ++seed) {
        const auto r = run(scenarios::attitude(30.0, seed));
        nees_sum += r.nees_sum;
        count += r.nees_count;
    }
    const double mean_nees = nees_sum / count;
    UNSCOPED_INFO("[report] mean NEES " << mean_nees << " (DoF 6)");
    // Expected value is DoF = 6. Means over disjoint 20-seed blocks ranged from
    // 5.1 to 6.9, so +/-40% catches a mis-scaled covariance without flaking.
    REQUIRE(mean_nees > 0.6 * 6);
    REQUIRE(mean_nees < 1.4 * 6);
}

TEST_CASE("Error-state update matches the analytic H and reset Jacobian", "[unit][eskf]") {
    // A large innovation gives a ~0.3 rad correction, where the reset Jacobian
    // J_r(d) is far from the identity.
    AttitudeBias<double> x;
    x.R.q = kalman::so3::exp(Eigen::Vector3d(0.4, -0.1, 0.7));
    x.bias = Eigen::Vector3d(0.01, 0.0, -0.02);
    ESKF::Cov P = ESKF::Cov::Identity() * 0.3;
    P(0, 3) = P(3, 0) = 0.05;
    const Eigen::Vector3d ref = Eigen::Vector3d(0.6, 0.0, 0.8);
    const Eigen::Matrix3d R = 0.01 * Eigen::Matrix3d::Identity();
    const Eigen::Vector3d z = kalman::so3::exp(Eigen::Vector3d(0.2, 0.3, -0.25)).conjugate() * ref;

    ESKF eskf(x, P);
    REQUIRE(eskf.update(scenarios::DirectionModel{ref}, z, R));

    // h(x [+] d) = Exp(-d) R^T ref ~ v + [v]x d, with v = R^T ref.
    const Eigen::Vector3d v = x.R.q.conjugate() * ref;
    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
    H.leftCols<3>() = kalman::so3::skew(v);
    const Eigen::Matrix3d S = H * P * H.transpose() + R;
    const Eigen::Matrix<double, 6, 3> K = P * H.transpose() * S.inverse();
    const Eigen::Matrix<double, 6, 1> d = K * (z - v);
    const ESKF::Cov IKH = ESKF::Cov::Identity() - K * H;
    const ESKF::Cov P_joseph = IKH * P * IKH.transpose() + K * R * K.transpose();

    // Reset: (x [+] e) [-] (x [+] d) = Log(Exp(-d) Exp(e)) has Jacobian J_r(d) at e = d.
    ESKF::Cov G = ESKF::Cov::Identity();
    G.topLeftCorner<3, 3>() = right_jacobian(d.head<3>());
    REQUIRE(d.head<3>().norm() > 0.2);

    const AttitudeBias<double> expected = x.boxplus(d);
    REQUIRE(eskf.state().boxminus(expected).norm() < 1e-12);
    REQUIRE((eskf.covariance() - G * P_joseph * G.transpose()).cwiseAbs().maxCoeff() < 1e-12);
}
