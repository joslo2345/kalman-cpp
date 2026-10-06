#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace kalman::detail {

// Cholesky factorization A = L L^T for small fixed-size matrices, such as an
// innovation covariance. Plain loops over compile-time bounds, which the
// compiler unrolls; much faster than Eigen::LLT's general blocked algorithm at
// these sizes, while keeping the positive-definiteness check that
// S.inverse() would lose.
template <typename Scalar, int M>
class SmallCholesky {
public:
    explicit SmallCholesky(const Eigen::Matrix<Scalar, M, M>& A) {
        using std::sqrt;
        L_.setZero();
        for (int j = 0; j < M; ++j) {
            Scalar d = A(j, j);
            for (int k = 0; k < j; ++k) d -= L_(j, k) * L_(j, k);
            if (!(d > Scalar(0))) return;  // also rejects NaN
            L_(j, j) = sqrt(d);
            const Scalar inv = Scalar(1) / L_(j, j);
            for (int i = j + 1; i < M; ++i) {
                Scalar s = A(i, j);
                for (int k = 0; k < j; ++k) s -= L_(i, k) * L_(j, k);
                L_(i, j) = s * inv;
            }
        }
        ok_ = true;
    }

    // False if A was not (numerically) positive-definite.
    bool ok() const { return ok_; }

    // Solves A X = B by forward and back substitution.
    template <int C>
    Eigen::Matrix<Scalar, M, C> solve(const Eigen::Matrix<Scalar, M, C>& B) const {
        Eigen::Matrix<Scalar, M, C> X = B;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < M; ++i) {
                Scalar s = X(i, c);
                for (int k = 0; k < i; ++k) s -= L_(i, k) * X(k, c);
                X(i, c) = s / L_(i, i);
            }
            for (int i = M - 1; i >= 0; --i) {
                Scalar s = X(i, c);
                for (int k = i + 1; k < M; ++k) s -= L_(k, i) * X(k, c);
                X(i, c) = s / L_(i, i);
            }
        }
        return X;
    }

    const Eigen::Matrix<Scalar, M, M>& matrixL() const { return L_; }

private:
    Eigen::Matrix<Scalar, M, M> L_;
    bool ok_ = false;
};

}  // namespace kalman::detail
