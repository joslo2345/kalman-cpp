#pragma once

#include <concepts>
#include <type_traits>

#include <Eigen/Dense>

#include "kalman/autodiff.hpp"

namespace kalman {

namespace detail {

template <typename Expr, int Rows, int Cols = 1>
concept HasShape = std::remove_cvref_t<Expr>::RowsAtCompileTime == Rows &&
                   std::remove_cvref_t<Expr>::ColsAtCompileTime == Cols;

// Shape check plus an exact scalar check. Needed for autodiff detection: Eigen's
// converting constructors would otherwise let a double-only model "accept" a
// vector of jets.
template <typename Expr, typename Scalar, int Rows, int Cols = 1>
concept HasShapeAndScalar =
    HasShape<Expr, Rows, Cols> && std::same_as<typename std::remove_cvref_t<Expr>::Scalar, Scalar>;

}  // namespace detail

// x_{k+1} = f(x_k, dt), evaluated with plain scalars.
template <typename M, int N, typename Scalar = double>
concept ProcessModel = requires(const M m, const Eigen::Matrix<Scalar, N, 1>& x, double dt) {
    { m.predict(x, dt) } -> detail::HasShape<N>;
};

// A process model that supplies its own Jacobian df/dx.
template <typename M, int N, typename Scalar = double>
concept AnalyticProcessModel =
    ProcessModel<M, N, Scalar> && requires(const M m, const Eigen::Matrix<Scalar, N, 1>& x, double dt) {
        { m.jacobian(x, dt) } -> detail::HasShape<N, N>;
    };

// A process model whose `predict` is templated on the scalar, so the EKF can
// differentiate it automatically.
template <typename M, int N, typename Scalar = double>
concept AutodiffProcessModel =
    ProcessModel<M, N, Scalar> &&
    requires(const M m, const Eigen::Matrix<Jet<Scalar, N>, N, 1>& x, double dt) {
        { m.predict(x, dt) } -> detail::HasShapeAndScalar<Jet<Scalar, N>, N>;
    };

// z = h(x), with state size N and measurement size Mz.
template <typename M, int N, int Mz, typename Scalar = double>
concept MeasurementModel = requires(const M m, const Eigen::Matrix<Scalar, N, 1>& x) {
    { m.measure(x) } -> detail::HasShape<Mz>;
};

template <typename M, int N, int Mz, typename Scalar = double>
concept AnalyticMeasurementModel =
    MeasurementModel<M, N, Mz, Scalar> && requires(const M m, const Eigen::Matrix<Scalar, N, 1>& x) {
        { m.jacobian(x) } -> detail::HasShape<Mz, N>;
    };

template <typename M, int N, int Mz, typename Scalar = double>
concept AutodiffMeasurementModel =
    MeasurementModel<M, N, Mz, Scalar> &&
    requires(const M m, const Eigen::Matrix<Jet<Scalar, N>, N, 1>& x) {
        { m.measure(x) } -> detail::HasShapeAndScalar<Jet<Scalar, N>, Mz>;
    };

// Optional hook for measurements that live on a manifold (e.g. angles), where
// the innovation is not a plain subtraction.
template <typename M, int Mz, typename Scalar = double>
concept HasResidual = requires(const M m, const Eigen::Matrix<Scalar, Mz, 1>& z) {
    { m.residual(z, z) } -> detail::HasShape<Mz>;
};

}  // namespace kalman
