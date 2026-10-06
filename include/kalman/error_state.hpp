#pragma once

#include <concepts>
#include <type_traits>

#include <Eigen/Dense>

#include "kalman/autodiff.hpp"
#include "kalman/concepts.hpp"
#include "kalman/detail/joseph_update.hpp"

namespace kalman {

// A state living on a manifold (e.g. an orientation), described by a class
// template S<T> so it can be instantiated with autodiff jets:
//
//   template <typename T> struct S {
//       static constexpr int DoF = ...;                    // tangent dimension
//       S boxplus(const Eigen::Matrix<T, DoF, 1>& d) const;  // x [+] d
//       Eigen::Matrix<T, DoF, 1> boxminus(const S& y) const; // x [-] y
//       template <typename U> S<U> cast() const;
//   };
//
// See kalman::SO3 for an example; product states (e.g. orientation + gyro bias)
// combine the pieces' operators block by block.
template <template <typename> class S, typename Scalar>
concept ManifoldState = requires(const S<Scalar> x, const Eigen::Matrix<Scalar, S<Scalar>::DoF, 1>& d) {
    { x.boxplus(d) } -> std::same_as<S<Scalar>>;
    { x.boxminus(x) } -> detail::HasShape<S<Scalar>::DoF>;
    { x.template cast<Jet<Scalar, S<Scalar>::DoF>>() } -> std::same_as<S<Jet<Scalar, S<Scalar>::DoF>>>;
};

// Error-state (multiplicative) extended Kalman filter on a manifold.
//
// The filter keeps a nominal state x on the manifold and a Gaussian over the
// error d in its tangent space (true state = x [+] d). Every Jacobian is
// computed by autodiff through the model and the state's [+]/[-] operators, so
// models and states must be templated on the scalar:
//
//   process:      F = d/dd [ f(x [+] d) [-] f(x) ]           at d = 0
//   measurement:  H = d/dd h(x [+] d)                        at d = 0
//   reset:        G = d/de [ (x [+] e) [-] (x [+] d_hat) ]   at e = d_hat
//
// After each update the estimated error d_hat is folded into x, and P is
// mapped into the tangent space of the new nominal state with G.
template <template <typename> class S, typename Scalar = double>
    requires ManifoldState<S, Scalar>
class ErrorStateKalmanFilter {
public:
    static constexpr int N = S<Scalar>::DoF;
    using State = S<Scalar>;
    using Error = Eigen::Matrix<Scalar, N, 1>;
    using Cov = Eigen::Matrix<Scalar, N, N>;

    ErrorStateKalmanFilter(const State& x0, const Cov& P0) : x_(x0), P_(P0) {}

    // f.predict(x, dt) must be templated on the scalar and return S<T>.
    template <typename Model>
    void predict(const Model& f, double dt, const Cov& Q) {
        static_assert(std::same_as<std::remove_cvref_t<decltype(f.predict(std::declval<const S<J>&>(), dt))>, S<J>>,
                      "process model needs predict(x, dt) templated on the scalar, returning the state type");

        const State x_next = f.predict(x_, dt);
        const S<J> moved = f.predict(perturbed(), dt);
        const Cov F = gradient(moved.boxminus(x_next.template cast<J>()));

        x_ = x_next;
        P_ = F * P_ * F.transpose() + Q;
        detail::symmetrize(P_);
    }

    // h.measure(x) must be templated on the scalar and return a fixed-size
    // vector. An optional h.residual(z, z_pred) handles wrapped quantities.
    // Returns false (and leaves the estimate untouched) if the innovation
    // covariance is not positive-definite.
    template <int Mz, typename Model>
        requires requires(const Model h, const State x) {
            { h.measure(x) } -> detail::HasShape<Mz>;
        }
    bool update(const Model& h, const Eigen::Matrix<Scalar, Mz, 1>& z, const Eigen::Matrix<Scalar, Mz, Mz>& R) {
        static_assert(requires(const S<J>& xj) {
            { h.measure(xj) } -> detail::HasShapeAndScalar<J, Mz>;
        }, "measurement model needs measure(x) templated on the scalar");

        const Eigen::Matrix<Scalar, Mz, 1> z_pred = h.measure(x_);
        const Eigen::Matrix<Scalar, Mz, N> H = gradient(h.measure(perturbed()).eval());

        Eigen::Matrix<Scalar, Mz, 1> y;
        if constexpr (HasResidual<Model, Mz, Scalar>) {
            y = h.residual(z, z_pred);
        } else {
            y = z - z_pred;
        }

        Error d = Error::Zero();
        Cov P = P_;
        Scalar nis;
        if (!detail::joseph_update(d, P, H, y, R, nis)) return false;

        // Fold the error into the nominal state and re-express P around it.
        const State x_new = x_.boxplus(d);
        Eigen::Matrix<J, N, 1> e;
        for (int i = 0; i < N; ++i) e(i) = J(d(i), i);
        const Cov G = gradient(x_.template cast<J>().boxplus(e).boxminus(x_new.template cast<J>()));

        x_ = x_new;
        P_ = G * P * G.transpose();
        detail::symmetrize(P_);
        nis_ = nis;
        return true;
    }

    const State& state() const { return x_; }
    const Cov& covariance() const { return P_; }

    // Normalized innovation squared from the last successful update.
    Scalar nis() const { return nis_; }

    void set_state(const State& x, const Cov& P) {
        x_ = x;
        P_ = P;
    }

private:
    using J = Jet<Scalar, N>;

    // x [+] d with d = 0 seeded as the N autodiff inputs.
    S<J> perturbed() const {
        Eigen::Matrix<J, N, 1> d;
        for (int i = 0; i < N; ++i) d(i) = J(Scalar(0), i);
        return x_.template cast<J>().boxplus(d);
    }

    template <int Rows>
    static Eigen::Matrix<Scalar, Rows, N> gradient(const Eigen::Matrix<J, Rows, 1>& y) {
        Eigen::Matrix<Scalar, Rows, N> out;
        for (int r = 0; r < Rows; ++r) out.row(r) = y(r).v.transpose();
        return out;
    }

    State x_;
    Cov P_;
    Scalar nis_ = Scalar(0);
};

}  // namespace kalman
