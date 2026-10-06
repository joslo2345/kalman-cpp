#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <kalman/kalman.hpp>

#include "scenarios/constant_velocity.hpp"
#include "scenarios/opencv_helpers.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("Linear KF matches OpenCV on 2D constant-velocity tracking", "[comparison]") {
    auto sc = scenarios::constant_velocity_2d(/*steps=*/1000, /*seed=*/42);

    kalman::LinearFilter<4, 2> ours(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);

    cv::KalmanFilter theirs(4, 2, 0, CV_64F);
    scenarios::configure_opencv(theirs, sc);

    for (const auto& z : sc.measurements) {
        ours.predict();
        ours.update(z);

        theirs.predict();
        cv::Mat est = theirs.correct(scenarios::to_cv(z));

        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(ours.state()(i), WithinAbs(est.at<double>(i), 1e-9));
        }
        const Eigen::Matrix4d P_cv = scenarios::from_cv<double, 4, 4>(theirs.errorCovPost);
        REQUIRE((ours.covariance() - P_cv).cwiseAbs().maxCoeff() < 1e-9);
    }
}
