#pragma once

// Eigen must come before OpenCV's Eigen bridge.
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

namespace scenarios {

template <typename Derived>
cv::Mat to_cv(const Eigen::MatrixBase<Derived>& m) {
    cv::Mat out;
    cv::eigen2cv(m.eval(), out);
    return out;
}

template <typename Scalar, int R, int C = 1>
Eigen::Matrix<Scalar, R, C> from_cv(const cv::Mat& m) {
    Eigen::Matrix<Scalar, R, C> out;
    cv::cv2eigen(m, out);
    return out;
}

// Copies a linear scenario (F, H, Q, R, x0, P0) into an OpenCV filter.
template <typename Scenario>
void configure_opencv(cv::KalmanFilter& kf, const Scenario& sc) {
    cv::eigen2cv(sc.F, kf.transitionMatrix);
    cv::eigen2cv(sc.H, kf.measurementMatrix);
    cv::eigen2cv(sc.Q, kf.processNoiseCov);
    cv::eigen2cv(sc.R, kf.measurementNoiseCov);
    cv::eigen2cv(sc.P0, kf.errorCovPost);
    cv::eigen2cv(sc.x0, kf.statePost);
}

}  // namespace scenarios
