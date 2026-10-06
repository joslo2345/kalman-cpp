#include <cmath>
#include <limits>
#include <random>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Cholesky>
#include <kalman/detail/small_cholesky.hpp>

namespace {

template <int M>
Eigen::Matrix<double, M, M> random_spd(std::mt19937& rng) {
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::Matrix<double, M, M> A;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j) A(i, j) = n(rng);
    return A * A.transpose() + 0.1 * Eigen::Matrix<double, M, M>::Identity();
}

}  // namespace

TEMPLATE_TEST_CASE_SIG("SmallCholesky matches Eigen::LLT", "[unit][cholesky]", ((int M), M), 1, 2, 3, 6, 15) {
    std::mt19937 rng(M);
    for (int trial = 0; trial < 50; ++trial) {
        const Eigen::Matrix<double, M, M> A = random_spd<M>(rng);
        const kalman::detail::SmallCholesky<double, M> ours(A);
        const Eigen::LLT<Eigen::Matrix<double, M, M>> ref(A);
        REQUIRE(ours.ok());
        const Eigen::Matrix<double, M, M> L_ref = ref.matrixL();
        REQUIRE((ours.matrixL() - L_ref).cwiseAbs().maxCoeff() < 1e-12 * (1 + L_ref.cwiseAbs().maxCoeff()));

        const Eigen::Matrix<double, M, 3> B = Eigen::Matrix<double, M, 3>::Random();
        REQUIRE((A * ours.template solve<3>(B) - B).cwiseAbs().maxCoeff() < 1e-10);
    }
}

TEST_CASE("SmallCholesky rejects matrices that are not positive-definite", "[unit][cholesky]") {
    using kalman::detail::SmallCholesky;
    REQUIRE_FALSE(SmallCholesky<double, 2>(Eigen::Matrix2d(Eigen::Vector2d(1.0, -1.0).asDiagonal())).ok());
    REQUIRE_FALSE(SmallCholesky<double, 2>(Eigen::Matrix2d::Zero()).ok());
    Eigen::Matrix2d indefinite;
    indefinite << 1.0, 2.0, 2.0, 1.0;  // eigenvalues 3 and -1
    REQUIRE_FALSE(SmallCholesky<double, 2>(indefinite).ok());
    Eigen::Matrix2d nan = Eigen::Matrix2d::Identity();
    nan(1, 1) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_FALSE(SmallCholesky<double, 2>(nan).ok());
}
