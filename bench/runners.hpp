#pragma once

// One runner per library and filter, shared by the timing, accuracy and
// allocation benchmarks so every library is driven the same way.

#include <cmath>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Dense>
// OpenCV's Eigen bridge must come after Eigen.
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/tracking/kalman_filters.hpp>
#include <opencv2/video/tracking.hpp>

#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman/UnscentedKalmanFilter.hpp>

#include <kalman/kalman.hpp>

#include "scenarios/range_bearing.hpp"
#include "scenarios/vectors.hpp"

namespace bench {

template <int N, typename Scalar = double>
struct Estimate {
    Eigen::Matrix<Scalar, N, 1> x;
    Eigen::Matrix<Scalar, N, N> P;
};

// ---------------------------------------------------------------- kalman-cpp

template <int N, int M, typename Scalar>
kalman::LinearFilter<N, M, Scalar> make_ours(const scenarios::Scenario<N, M, Scalar>& sc) {
    return kalman::LinearFilter<N, M, Scalar>(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
}

template <int N, int M, typename Scalar>
kalman::SquareRootLinearFilter<N, M, Scalar> make_ours_sqrt(const scenarios::Scenario<N, M, Scalar>& sc) {
    return kalman::SquareRootLinearFilter<N, M, Scalar>(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
}

// ---------------------------------------------------------------- OpenCV

template <int N, int M, typename Scalar>
cv::KalmanFilter make_opencv(const scenarios::Scenario<N, M, Scalar>& sc) {
    cv::KalmanFilter kf(N, M, 0, std::is_same_v<Scalar, float> ? CV_32F : CV_64F);
    cv::eigen2cv(sc.F, kf.transitionMatrix);
    cv::eigen2cv(sc.H, kf.measurementMatrix);
    cv::eigen2cv(sc.Q, kf.processNoiseCov);
    cv::eigen2cv(sc.R, kf.measurementNoiseCov);
    cv::eigen2cv(sc.P0, kf.errorCovPost);
    cv::eigen2cv(sc.x0, kf.statePost);
    return kf;
}

template <typename Scalar, int R, int C = 1>
Eigen::Matrix<Scalar, R, C> from_cv(const cv::Mat& m) {
    Eigen::Matrix<Scalar, R, C> out;
    cv::cv2eigen(m, out);
    return out;
}

// Converted once, outside any timed region.
template <int M, typename Scalar>
std::vector<cv::Mat> to_cv(const std::vector<Eigen::Matrix<Scalar, M, 1>>& zs) {
    std::vector<cv::Mat> out;
    out.reserve(zs.size());
    for (const auto& z : zs) {
        cv::Mat m;
        cv::eigen2cv(z, m);
        out.push_back(m);
    }
    return out;
}

// ---------------------------------------------------------------- mherb/kalman
//
// mherb/kalman has no plain linear KF; its EKF with constant Jacobians is one.

template <int N, typename Scalar>
using MState = Kalman::Vector<Scalar, N>;

template <int N, typename Scalar>
class MherbLinearSystem : public Kalman::LinearizedSystemModel<MState<N, Scalar>> {
public:
    using S = MState<N, Scalar>;
    MherbLinearSystem(const Eigen::Matrix<Scalar, N, N>& F, const Eigen::Matrix<Scalar, N, N>& Q) {
        this->F = F;
        this->setCovariance(Q);
    }
    S f(const S& x, const typename MherbLinearSystem::Control&) const override { return this->F * x; }
};

template <int N, int M, typename Scalar>
class MherbLinearMeasurement : public Kalman::LinearizedMeasurementModel<MState<N, Scalar>, MState<M, Scalar>> {
public:
    MherbLinearMeasurement(const Eigen::Matrix<Scalar, M, N>& H, const Eigen::Matrix<Scalar, M, M>& R) {
        this->H = H;
        this->setCovariance(R);
    }
    MState<M, Scalar> h(const MState<N, Scalar>& x) const override { return this->H * x; }
};

template <int N, int M, typename Scalar>
struct MherbLinear {
    Kalman::ExtendedKalmanFilter<MState<N, Scalar>> kf;
    MherbLinearSystem<N, Scalar> sys;
    MherbLinearMeasurement<N, M, Scalar> meas;

    explicit MherbLinear(const scenarios::Scenario<N, M, Scalar>& sc) : sys(sc.F, sc.Q), meas(sc.H, sc.R) {
        kf.init(MState<N, Scalar>(sc.x0));
        kf.setCovariance(sc.P0);
    }
    void step(const MState<M, Scalar>& z) {
        kf.predict(sys);
        kf.update(meas, z);
    }
};

// Range-bearing models for mherb's EKF and UKF on S3.
using MState4 = MState<4, double>;
using MMeas2 = MState<2, double>;

class MherbCV : public Kalman::LinearizedSystemModel<MState4> {
public:
    MherbCV(double dt, const Eigen::Matrix4d& Q) {
        this->F = scenarios::cv_transition(dt);
        this->setCovariance(Q);
    }
    MState4 f(const MState4& x, const Control&) const override { return this->F * x; }
};

// mherb/kalman and OpenCV's UKF compute innovations and sigma-point means as
// plain differences and averages, with no hook for angles. Near +/-pi that
// turns a small bearing error into ~2*pi. The workaround a user would apply:
// before each update, set a reference angle (the predicted bearing), have
// h(x) return bearings unwrapped near it, and unwrap z the same way. Bearings
// are then continuous for every sigma point and for the innovation.
inline double unwrap_near(double angle, double ref) { return ref + scenarios::wrap_angle(angle - ref); }

inline Eigen::Vector2d range_bearing_near(const Eigen::Vector4d& x, double ref) {
    Eigen::Vector2d z = scenarios::RangeBearingModel{}.measure(x);
    z(1) = unwrap_near(z(1), ref);
    return z;
}

class MherbRangeBearing : public Kalman::LinearizedMeasurementModel<MState4, MMeas2> {
public:
    explicit MherbRangeBearing(const Eigen::Matrix2d& R) { this->setCovariance(R); }
    MMeas2 h(const MState4& x) const override { return MMeas2(range_bearing_near(x, ref)); }
    void updateJacobians(const MState4& x) override { this->H = scenarios::RangeBearingModel{}.analytic_jacobian(x); }
    double ref = 0.0;  // reference bearing, set before each update
};

template <typename Filter>
std::vector<Estimate<4>> run_mherb_range_bearing(const scenarios::MultiSeedScenario& sc, std::size_t seed) {
    Filter kf;
    MherbCV sys(sc.dt, sc.Q);
    MherbRangeBearing meas(sc.R);
    kf.init(MState4(sc.x0));
    kf.setCovariance(sc.P0);
    std::vector<Estimate<4>> out;
    for (const auto& z : sc.zs[seed]) {
        kf.predict(sys);
        const Eigen::Vector4d x_pred = kf.getState();
        meas.ref = std::atan2(x_pred(1), x_pred(0));
        kf.update(meas, MMeas2(Eigen::Vector2d(z(0), unwrap_near(z(1), meas.ref))));
        out.push_back({kf.getState(), kf.getCovariance()});
    }
    return out;
}

// ---------------------------------------------------------------- OpenCV, nonlinear (S3)

// OpenCV has no EKF class. The usual way to build one on cv::KalmanFilter:
// re-linearize measurementMatrix at the prediction and pass the
// pseudo-measurement z - h(x) + H x, so its linear residual equals the EKF
// innovation (validated against our EKF in tests/comparison).
inline std::vector<Estimate<4>> run_opencv_ekf_range_bearing(const scenarios::MultiSeedScenario& sc,
                                                             std::size_t seed) {
    const scenarios::RangeBearingModel h;
    cv::KalmanFilter kf(4, 2, 0, CV_64F);
    cv::eigen2cv(sc.F, kf.transitionMatrix);
    cv::eigen2cv(sc.Q, kf.processNoiseCov);
    cv::eigen2cv(sc.R, kf.measurementNoiseCov);
    cv::eigen2cv(sc.P0, kf.errorCovPost);
    cv::eigen2cv(sc.x0, kf.statePost);
    std::vector<Estimate<4>> out;
    for (const auto& z : sc.zs[seed]) {
        const Eigen::Vector4d x_pred = from_cv<double, 4>(kf.predict());
        const Eigen::Matrix<double, 2, 4> H = h.analytic_jacobian(x_pred);
        cv::eigen2cv(H, kf.measurementMatrix);
        const Eigen::Vector2d pseudo = h.residual(z, h.measure(x_pred)) + H * x_pred;
        cv::Mat zm;
        cv::eigen2cv(pseudo, zm);
        kf.correct(zm);
        out.push_back({from_cv<double, 4>(kf.statePost), from_cv<double, 4, 4>(kf.errorCovPost)});
    }
    return out;
}

// The UKF from OpenCV's contrib tracking module (cv::detail in OpenCV 5).
// Noise enters additively through v_k and n_k.
class OpencvRangeBearingModel : public cv::detail::tracking::UkfSystemModel {
public:
    explicit OpencvRangeBearingModel(double dt) : F_(scenarios::cv_transition(dt)) {}
    void stateConversionFunction(const cv::Mat& x_k, const cv::Mat&, const cv::Mat& v_k, cv::Mat& x_next) override {
        const Eigen::Vector4d x = from_cv<double, 4>(x_k);
        cv::eigen2cv(Eigen::Vector4d(F_ * x), x_next);
        x_next += v_k;
    }
    void measurementFunction(const cv::Mat& x_k, const cv::Mat& n_k, cv::Mat& z_k) override {
        cv::eigen2cv(range_bearing_near(from_cv<double, 4>(x_k), ref), z_k);
        z_k += n_k;
    }
    double ref = 0.0;  // reference bearing, set before each correct()

private:
    Eigen::Matrix4d F_;
};

inline std::vector<Estimate<4>> run_opencv_ukf_range_bearing(const scenarios::MultiSeedScenario& sc,
                                                             std::size_t seed) {
    auto model = cv::makePtr<OpencvRangeBearingModel>(sc.dt);
    cv::detail::tracking::UnscentedKalmanFilterParams params(4, 2, 0, 0.0, 0.0, model, CV_64F);
    cv::eigen2cv(sc.Q, params.processNoiseCov);
    cv::eigen2cv(sc.R, params.measurementNoiseCov);
    cv::eigen2cv(sc.x0, params.stateInit);
    cv::eigen2cv(sc.P0, params.errorCovInit);
    // Same sigma-point parameters as kalman-cpp and mherb/kalman (OpenCV's
    // default alpha is 1e-3).
    params.alpha = 1.0;
    params.beta = 2.0;
    params.k = 0.0;
    auto ukf = cv::detail::tracking::createUnscentedKalmanFilter(params);

    std::vector<Estimate<4>> out;
    for (const auto& z : sc.zs[seed]) {
        const Eigen::Vector4d x_pred = from_cv<double, 4>(ukf->predict());
        model->ref = std::atan2(x_pred(1), x_pred(0));
        cv::Mat zm;
        cv::eigen2cv(Eigen::Vector2d(z(0), unwrap_near(z(1), model->ref)), zm);
        ukf->correct(zm);
        out.push_back({from_cv<double, 4>(ukf->getState()), from_cv<double, 4, 4>(ukf->getErrorCov())});
    }
    return out;
}

// ---------------------------------------------------------------- naive

// Textbook filter with the short-form covariance update P = (I - K H) P.
template <int N, int M, typename Scalar>
struct NaiveFilter {
    Eigen::Matrix<Scalar, N, N> F, Q, P;
    Eigen::Matrix<Scalar, M, N> H;
    Eigen::Matrix<Scalar, M, M> R;
    Eigen::Matrix<Scalar, N, 1> x;

    explicit NaiveFilter(const scenarios::Scenario<N, M, Scalar>& sc)
        : F(sc.F), Q(sc.Q), P(sc.P0), H(sc.H), R(sc.R), x(sc.x0) {}

    void step(const Eigen::Matrix<Scalar, M, 1>& z) {
        x = F * x;
        P = F * P * F.transpose() + Q;
        const Eigen::Matrix<Scalar, M, M> S = H * P * H.transpose() + R;
        const Eigen::Matrix<Scalar, N, M> K = P * H.transpose() * S.inverse();
        x += K * (z - H * x);
        P = (Eigen::Matrix<Scalar, N, N>::Identity() - K * H) * P;
    }
};

// ---------------------------------------------------------------- checks

// Symmetric (relative to scale) and positive-definite.
template <typename Mat>
bool is_spd(const Mat& P) {
    using S = typename Mat::Scalar;
    if (!P.allFinite()) return false;
    if ((P - P.transpose()).cwiseAbs().maxCoeff() > S(1e-4) * P.cwiseAbs().maxCoeff()) return false;
    return Eigen::LLT<Mat>(P).info() == Eigen::Success;
}

}  // namespace bench
