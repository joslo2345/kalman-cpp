#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

namespace kalman {

// Forward-mode dual number: a value plus its gradient with respect to N inputs.
//
// Models opt into autodiff by being templated on the scalar type and calling
// math functions unqualified (`using std::sin; sin(x(0))`) so that argument-
// dependent lookup finds the overloads below when the scalar is a Jet.
template <typename T, int N>
struct Jet {
    using Derivative = Eigen::Matrix<T, N, 1>;

    T a{};
    Derivative v = Derivative::Zero();

    Jet() = default;
    Jet(const T& value) : a(value) {}  // NOLINT(google-explicit-constructor)
    Jet(const T& value, const Derivative& grad) : a(value), v(grad) {}
    // Seeds the i-th input variable: derivative is the i-th unit vector.
    Jet(const T& value, int i) : a(value) { v(i) = T(1); }

    Jet& operator+=(const Jet& b) { a += b.a; v += b.v; return *this; }
    Jet& operator-=(const Jet& b) { a -= b.a; v -= b.v; return *this; }
    Jet& operator*=(const Jet& b) { return *this = *this * b; }
    Jet& operator/=(const Jet& b) { return *this = *this / b; }
    Jet& operator+=(const T& s) { a += s; return *this; }
    Jet& operator-=(const T& s) { a -= s; return *this; }
    Jet& operator*=(const T& s) { a *= s; v *= s; return *this; }
    Jet& operator/=(const T& s) { a /= s; v /= s; return *this; }

    // Hidden friends, so mixed Jet/scalar expressions work with implicit
    // conversions (e.g. `x(0) * 2` with T = double).
    friend Jet operator+(const Jet& f) { return f; }
    friend Jet operator-(const Jet& f) { return Jet(-f.a, -f.v); }

    friend Jet operator+(const Jet& f, const Jet& g) { return Jet(f.a + g.a, f.v + g.v); }
    friend Jet operator+(const Jet& f, const T& s) { return Jet(f.a + s, f.v); }
    friend Jet operator+(const T& s, const Jet& f) { return Jet(f.a + s, f.v); }

    friend Jet operator-(const Jet& f, const Jet& g) { return Jet(f.a - g.a, f.v - g.v); }
    friend Jet operator-(const Jet& f, const T& s) { return Jet(f.a - s, f.v); }
    friend Jet operator-(const T& s, const Jet& f) { return Jet(s - f.a, -f.v); }

    friend Jet operator*(const Jet& f, const Jet& g) { return Jet(f.a * g.a, f.a * g.v + f.v * g.a); }
    friend Jet operator*(const Jet& f, const T& s) { return Jet(f.a * s, f.v * s); }
    friend Jet operator*(const T& s, const Jet& f) { return Jet(f.a * s, f.v * s); }

    friend Jet operator/(const Jet& f, const Jet& g) {
        const T inv = T(1) / g.a;
        const T q = f.a * inv;
        return Jet(q, (f.v - q * g.v) * inv);
    }
    friend Jet operator/(const Jet& f, const T& s) { return Jet(f.a / s, f.v / s); }
    friend Jet operator/(const T& s, const Jet& g) {
        const T q = s / g.a;
        return Jet(q, g.v * (-q / g.a));
    }

    // Comparisons use the value only.
    friend bool operator==(const Jet& f, const Jet& g) { return f.a == g.a; }
    friend bool operator!=(const Jet& f, const Jet& g) { return f.a != g.a; }
    friend bool operator<(const Jet& f, const Jet& g) { return f.a < g.a; }
    friend bool operator<=(const Jet& f, const Jet& g) { return f.a <= g.a; }
    friend bool operator>(const Jet& f, const Jet& g) { return f.a > g.a; }
    friend bool operator>=(const Jet& f, const Jet& g) { return f.a >= g.a; }
};

template <typename T>
struct is_jet : std::false_type {};
template <typename T, int N>
struct is_jet<Jet<T, N>> : std::true_type {};

// ---- Elementary functions (chain rule: f(a + v e) = f(a) + f'(a) v e) ----

template <typename T, int N>
Jet<T, N> chain(const Jet<T, N>& f, const T& value, const T& derivative) {
    return Jet<T, N>(value, f.v * derivative);
}

template <typename T, int N>
Jet<T, N> abs(const Jet<T, N>& f) { return f.a < T(0) ? -f : f; }
template <typename T, int N>
Jet<T, N> sqrt(const Jet<T, N>& f) {
    using std::sqrt;
    const T s = sqrt(f.a);
    return chain(f, s, T(0.5) / s);
}
template <typename T, int N>
Jet<T, N> exp(const Jet<T, N>& f) {
    using std::exp;
    const T e = exp(f.a);
    return chain(f, e, e);
}
template <typename T, int N>
Jet<T, N> log(const Jet<T, N>& f) {
    using std::log;
    return chain(f, log(f.a), T(1) / f.a);
}
template <typename T, int N>
Jet<T, N> sin(const Jet<T, N>& f) {
    using std::cos;
    using std::sin;
    return chain(f, sin(f.a), cos(f.a));
}
template <typename T, int N>
Jet<T, N> cos(const Jet<T, N>& f) {
    using std::cos;
    using std::sin;
    return chain(f, cos(f.a), -sin(f.a));
}
template <typename T, int N>
Jet<T, N> tan(const Jet<T, N>& f) {
    using std::tan;
    const T t = tan(f.a);
    return chain(f, t, T(1) + t * t);
}
template <typename T, int N>
Jet<T, N> asin(const Jet<T, N>& f) {
    using std::asin;
    using std::sqrt;
    return chain(f, asin(f.a), T(1) / sqrt(T(1) - f.a * f.a));
}
template <typename T, int N>
Jet<T, N> acos(const Jet<T, N>& f) {
    using std::acos;
    using std::sqrt;
    return chain(f, acos(f.a), -T(1) / sqrt(T(1) - f.a * f.a));
}
template <typename T, int N>
Jet<T, N> atan(const Jet<T, N>& f) {
    using std::atan;
    return chain(f, atan(f.a), T(1) / (T(1) + f.a * f.a));
}
template <typename T, int N>
Jet<T, N> sinh(const Jet<T, N>& f) {
    using std::cosh;
    using std::sinh;
    return chain(f, sinh(f.a), cosh(f.a));
}
template <typename T, int N>
Jet<T, N> cosh(const Jet<T, N>& f) {
    using std::cosh;
    using std::sinh;
    return chain(f, cosh(f.a), sinh(f.a));
}
template <typename T, int N>
Jet<T, N> tanh(const Jet<T, N>& f) {
    using std::tanh;
    const T t = tanh(f.a);
    return chain(f, t, T(1) - t * t);
}
template <typename T, int N>
Jet<T, N> atan2(const Jet<T, N>& y, const Jet<T, N>& x) {
    using std::atan2;
    const T r2 = x.a * x.a + y.a * y.a;
    return Jet<T, N>(atan2(y.a, x.a), (x.a * y.v - y.a * x.v) / r2);
}
template <typename T, int N>
Jet<T, N> hypot(const Jet<T, N>& x, const Jet<T, N>& y) {
    using std::hypot;
    const T h = hypot(x.a, y.a);
    return Jet<T, N>(h, (x.a * x.v + y.a * y.v) / h);
}
template <typename T, int N>
Jet<T, N> pow(const Jet<T, N>& f, const T& p) {
    using std::pow;
    return chain(f, pow(f.a, p), p * pow(f.a, p - T(1)));
}
template <typename T, int N>
Jet<T, N> pow(const T& b, const Jet<T, N>& g) {
    using std::log;
    using std::pow;
    const T value = pow(b, g.a);
    return chain(g, value, value * log(b));
}
template <typename T, int N>
Jet<T, N> pow(const Jet<T, N>& f, const Jet<T, N>& g) {
    using std::log;
    using std::pow;
    const T value = pow(f.a, g.a);
    return Jet<T, N>(value, value * (g.a / f.a * f.v + log(f.a) * g.v));
}

template <typename T, int N>
bool isfinite(const Jet<T, N>& f) {
    using std::isfinite;
    return isfinite(f.a) && f.v.allFinite();
}
template <typename T, int N>
bool isnan(const Jet<T, N>& f) {
    using std::isnan;
    return isnan(f.a) || f.v.hasNaN();
}

// ---- Jacobians ----

namespace detail {

template <typename F, typename Vec>
auto call_model(const F& f, const Vec& x) {
    if constexpr (requires { f(x); }) {
        return f(x).eval();
    } else {
        return f.measure(x).eval();
    }
}

}  // namespace detail

// Evaluates y = f(x) and its Jacobian dy/dx in one forward pass. `f` is either a
// callable or a measurement model with a templated `measure(x)`.
template <typename F, typename Scalar, int N>
auto value_and_jacobian(const F& f, const Eigen::Matrix<Scalar, N, 1>& x) {
    using J = Jet<Scalar, N>;
    Eigen::Matrix<J, N, 1> xj;
    for (int i = 0; i < N; ++i) xj(i) = J(x(i), i);

    const auto yj = detail::call_model(f, xj);
    constexpr int M = std::remove_cvref_t<decltype(yj)>::RowsAtCompileTime;
    static_assert(M != Eigen::Dynamic, "model output must have a fixed size");

    std::pair<Eigen::Matrix<Scalar, M, 1>, Eigen::Matrix<Scalar, M, N>> out;
    for (int r = 0; r < M; ++r) {
        out.first(r) = yj(r).a;
        out.second.row(r) = yj(r).v.transpose();
    }
    return out;
}

template <typename F, typename Scalar, int N>
auto jacobian(const F& f, const Eigen::Matrix<Scalar, N, 1>& x) {
    return value_and_jacobian(f, x).second;
}

}  // namespace kalman

// ---- Eigen integration, so Eigen::Matrix<Jet, ...> works ----
namespace Eigen {

template <typename T, int N>
struct NumTraits<kalman::Jet<T, N>> : GenericNumTraits<kalman::Jet<T, N>> {
    using Real = kalman::Jet<T, N>;
    using NonInteger = kalman::Jet<T, N>;
    using Nested = kalman::Jet<T, N>;
    using Literal = kalman::Jet<T, N>;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 3,
    };

    static inline Real epsilon() { return Real(std::numeric_limits<T>::epsilon()); }
    static inline Real dummy_precision() { return Real(NumTraits<T>::dummy_precision()); }
    static inline Real highest() { return Real(std::numeric_limits<T>::max()); }
    static inline Real lowest() { return Real(std::numeric_limits<T>::lowest()); }
    static inline int digits10() { return NumTraits<T>::digits10(); }
};

template <typename T, int N, typename BinaryOp>
struct ScalarBinaryOpTraits<kalman::Jet<T, N>, T, BinaryOp> {
    using ReturnType = kalman::Jet<T, N>;
};
template <typename T, int N, typename BinaryOp>
struct ScalarBinaryOpTraits<T, kalman::Jet<T, N>, BinaryOp> {
    using ReturnType = kalman::Jet<T, N>;
};

}  // namespace Eigen
