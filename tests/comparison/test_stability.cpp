#include <catch2/catch_test_macros.hpp>
#include <Eigen/Cholesky>
#include <kalman/kalman.hpp>

#include "scenarios/ill_conditioned.hpp"
#include "scenarios/opencv_helpers.hpp"

namespace {

// Symmetry is checked relative to the matrix's scale: an absolute tolerance
// would flag ordinary float round-off on large covariances as a failure.
template <typename Mat>
bool is_spd(const Mat& P) {
    if ((P - P.transpose()).cwiseAbs().maxCoeff() > 1e-4f * P.cwiseAbs().maxCoeff()) return false;
    return Eigen::LLT<Mat>(P).info() == Eigen::Success;
}

}  // namespace

// A vague prior (variance 1e6) meets very precise measurements (variance
// 1e-6). In float, any covariance-form filter loses information at the first
// predict: the position variance (~1e-6) added to ~1e4 rounds away, so the
// prior is singular to working precision and the next posterior is wrong.
// OpenCV's short-form update collapses P to zero; our Joseph-form KF gets
// tiny eigenvalues of either sign depending on rounding (it is reported, not
// required). The square-root KF carries the Cholesky factor, where a 1e-3
// standard deviation sits next to 1e2 without loss, and must stay SPD.
TEST_CASE("Covariance stays SPD over 1M float steps", "[comparison][stability]") {
    auto sc = scenarios::ill_conditioned<float>(/*seed=*/7);

    kalman::SquareRootLinearFilter<4, 2, float> ours(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    kalman::LinearFilter<4, 2, float> joseph(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    cv::KalmanFilter theirs(4, 2, 0, CV_32F);
    scenarios::configure_opencv(theirs, sc);

    long joseph_failed_at = -1, opencv_failed_at = -1;
    for (long k = 0; k < 1'000'000; ++k) {
        auto z = sc.measure(k);

        ours.predict();
        REQUIRE(ours.update(z));
        REQUIRE(is_spd(ours.covariance()));

        if (joseph_failed_at < 0) {
            joseph.predict();
            if (!joseph.update(z) || !is_spd(joseph.covariance())) joseph_failed_at = k;
        }
        if (opencv_failed_at < 0) {
            theirs.predict();
            theirs.correct(scenarios::to_cv(z));
            if (!is_spd(scenarios::from_cv<float, 4, 4>(theirs.errorCovPost))) opencv_failed_at = k;
        }
    }
    UNSCOPED_INFO("[report] Joseph-form KF lost SPD at step " << joseph_failed_at << ", OpenCV at step "
                                                              << opencv_failed_at << " (-1 = never)");
    SUCCEED();  // attaches the report to an assertion so Catch2 prints it with -s
}
