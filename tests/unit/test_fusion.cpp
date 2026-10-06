#include <algorithm>
#include <cmath>
#include <random>
#include <variant>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"
#include "scenarios/range_bearing.hpp"

namespace {

struct PositionModel {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 4, 1>& x) const {
        return x.template head<2>();
    }
};

// A 1-D sensor (x position only), to mix measurement sizes.
struct XModel {
    template <typename T>
    Eigen::Matrix<T, 1, 1> measure(const Eigen::Matrix<T, 4, 1>& x) const {
        return x.template head<1>();
    }
};

Eigen::Vector4d truth_at(double t) { return {-25.0 + 5.0 * t, 2.0 + 0.5 * std::sin(t), 5.0, 0.5 * std::cos(t)}; }

// One measurement from one of three sensors running at different rates.
struct Meas {
    double t;
    double arrival;
    int sensor;  // 0: position @ 10 Hz, 1: range-bearing @ ~3.3 Hz, 2: x-only @ 7 Hz
    Eigen::VectorXd z;
};

const Eigen::Matrix2d R_pos = 0.25 * Eigen::Matrix2d::Identity();
const Eigen::Matrix2d R_rb = Eigen::Vector2d(0.04, 0.0025).asDiagonal();
const Eigen::Matrix<double, 1, 1> R_x = Eigen::Matrix<double, 1, 1>(0.09);

std::vector<Meas> multi_rate_measurements(double duration, double max_delay, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> delay(0.0, max_delay);
    std::vector<Meas> out;
    // Offsets keep timestamps of different sensors distinct.
    for (int i = 0; 0.05 + 0.1 * i < duration; ++i) {
        const double t = 0.05 + 0.1 * i;
        const Eigen::Vector4d x = truth_at(t);
        out.push_back({t, 0, 0, Eigen::Vector2d(x(0) + 0.5 * n(rng), x(1) + 0.5 * n(rng))});
    }
    for (int i = 0; 0.013 + 0.3 * i < duration; ++i) {
        const double t = 0.013 + 0.3 * i;
        Eigen::Vector2d z = scenarios::RangeBearingModel{}.measure(truth_at(t));
        z += Eigen::Vector2d(0.2 * n(rng), 0.05 * n(rng));
        z(1) = scenarios::wrap_angle(z(1));
        out.push_back({t, 0, 1, z});
    }
    for (int i = 0; 0.031 + i / 7.0 < duration; ++i) {
        const double t = 0.031 + i / 7.0;
        out.push_back({t, 0, 2, Eigen::Matrix<double, 1, 1>(truth_at(t)(0) + 0.3 * n(rng))});
    }
    for (auto& m : out) m.arrival = m.t + delay(rng);
    return out;
}

template <typename Fusion>
kalman::FusionResult feed(Fusion& fusion, const Meas& m) {
    switch (m.sensor) {
        case 0:
            return fusion.add(m.t, PositionModel{}, Eigen::Vector2d(m.z), R_pos);
        case 1:
            return fusion.add(m.t, scenarios::RangeBearingModel{}, Eigen::Vector2d(m.z), R_rb);
        default:
            return fusion.add(m.t, XModel{}, Eigen::Matrix<double, 1, 1>(m.z), R_x);
    }
}

auto process_noise = [](double dt) { return scenarios::cv_process_noise(dt, 0.5); };

template <typename Filter>
Filter make_filter() {
    return Filter(Eigen::Vector4d(-25.0, 2.0, 5.0, 0.0), Eigen::Vector4d(4.0, 4.0, 1.0, 1.0).asDiagonal());
}

// Reference: a plain loop over measurements sorted by timestamp.
template <typename Filter>
Filter process_in_order(std::vector<Meas> ms) {
    std::sort(ms.begin(), ms.end(), [](const Meas& a, const Meas& b) { return a.t < b.t; });
    Filter f = make_filter<Filter>();
    double t = 0.0;
    for (const auto& m : ms) {
        f.predict(scenarios::ConstantVelocity2D{}, m.t - t, process_noise(m.t - t));
        t = m.t;
        bool ok = false;
        switch (m.sensor) {
            case 0:
                ok = f.update(PositionModel{}, Eigen::Vector2d(m.z), R_pos);
                break;
            case 1:
                ok = f.update(scenarios::RangeBearingModel{}, Eigen::Vector2d(m.z), R_rb);
                break;
            default:
                ok = f.update(XModel{}, Eigen::Matrix<double, 1, 1>(m.z), R_x);
                break;
        }
        REQUIRE(ok);
    }
    return f;
}

template <typename Filter>
using Fusion = kalman::AsyncFusion<Filter, scenarios::ConstantVelocity2D>;

}  // namespace

TEMPLATE_TEST_CASE("Out-of-sequence measurements give the same result as in-order processing", "[unit][fusion]",
                   kalman::ExtendedKalmanFilter<4>, kalman::UnscentedKalmanFilter<4>,
                   kalman::SquareRootUnscentedKalmanFilter<4>) {
    auto ms = multi_rate_measurements(/*duration=*/10.0, /*max_delay=*/0.5, /*seed=*/4);
    const TestType reference = process_in_order<TestType>(ms);

    std::sort(ms.begin(), ms.end(), [](const Meas& a, const Meas& b) { return a.arrival < b.arrival; });
    Fusion<TestType> fusion(make_filter<TestType>(), 0.0, {}, process_noise, /*horizon=*/1.0);
    int reordered = 0;
    for (const auto& m : ms) {
        const auto result = feed(fusion, m);
        REQUIRE((result == kalman::FusionResult::applied || result == kalman::FusionResult::reordered));
        reordered += result == kalman::FusionResult::reordered;
    }
    UNSCOPED_INFO("[report] " << reordered << " of " << ms.size() << " measurements arrived out of order");
    CHECK(reordered > 50);  // the out-of-sequence path really was exercised

    REQUIRE((fusion.state() - reference.state()).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE((fusion.covariance() - reference.covariance()).cwiseAbs().maxCoeff() < 1e-12);
    const Eigen::Vector4d final_state = fusion.state();
    const Eigen::Vector4d final_truth = truth_at(fusion.time());
    REQUIRE((final_state.head<2>() - final_truth.head<2>()).norm() < 1.0);
}

TEST_CASE("Measurements older than the horizon are rejected", "[unit][fusion]") {
    Fusion<kalman::ExtendedKalmanFilter<4>> fusion(make_filter<kalman::ExtendedKalmanFilter<4>>(), 0.0, {},
                                                   process_noise, /*horizon=*/1.0);
    for (int i = 1; i < 50; ++i) {
        const double t = 0.1 * i;
        REQUIRE(fusion.add(t, PositionModel{}, Eigen::Vector2d(truth_at(t).head<2>()), R_pos) ==
                kalman::FusionResult::applied);
    }
    const Eigen::Vector4d before = fusion.state();
    REQUIRE(fusion.add(2.0, PositionModel{}, Eigen::Vector2d(0.0, 0.0), R_pos) == kalman::FusionResult::too_old);
    REQUIRE(fusion.state() == before);

    // The buffer only holds what fits in the horizon (about 10 measurements at 10 Hz).
    REQUIRE(fusion.buffered() <= 11);
    // Just inside the horizon is still accepted.
    REQUIRE(fusion.add(4.5, PositionModel{}, Eigen::Vector2d(truth_at(4.5).head<2>()), R_pos) ==
            kalman::FusionResult::reordered);
}

TEST_CASE("A rejected update leaves the fusion state untouched", "[unit][fusion]") {
    Fusion<kalman::UnscentedKalmanFilter<4>> fusion(make_filter<kalman::UnscentedKalmanFilter<4>>(), 0.0, {},
                                                    process_noise, 1.0);
    REQUIRE(fusion.add(0.1, PositionModel{}, Eigen::Vector2d(-24.5, 2.0), R_pos) == kalman::FusionResult::applied);
    REQUIRE(fusion.add(0.3, PositionModel{}, Eigen::Vector2d(-23.5, 2.1), R_pos) == kalman::FusionResult::applied);
    const Eigen::Vector4d before = fusion.state();
    const double t_before = fusion.time();
    const std::size_t n_before = fusion.buffered();

    const Eigen::Matrix2d bad_R = -100.0 * Eigen::Matrix2d::Identity();
    REQUIRE(fusion.add(0.2, PositionModel{}, Eigen::Vector2d(-24.0, 2.0), bad_R) ==
            kalman::FusionResult::update_failed);
    REQUIRE(fusion.add(0.4, PositionModel{}, Eigen::Vector2d(-23.0, 2.0), bad_R) ==
            kalman::FusionResult::update_failed);
    REQUIRE(fusion.state() == before);
    REQUIRE(fusion.time() == t_before);
    REQUIRE(fusion.buffered() == n_before);
}

TEST_CASE("predicted() looks ahead without changing the engine", "[unit][fusion]") {
    using EKF = kalman::ExtendedKalmanFilter<4>;
    Fusion<EKF> fusion(make_filter<EKF>(), 0.0, {}, process_noise, 1.0);
    REQUIRE(fusion.add(0.1, PositionModel{}, Eigen::Vector2d(-24.5, 2.0), R_pos) == kalman::FusionResult::applied);

    const EKF ahead = fusion.predicted(0.6);
    EKF manual = fusion.filter();
    manual.predict(scenarios::ConstantVelocity2D{}, 0.5, process_noise(0.5));
    REQUIRE((ahead.state() - manual.state()).cwiseAbs().maxCoeff() < 1e-15);
    REQUIRE(fusion.time() == 0.1);
    REQUIRE_THROWS_AS(fusion.predicted(0.05), std::invalid_argument);
}
