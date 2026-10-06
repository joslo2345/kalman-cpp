#pragma once

#include <concepts>

#include <Eigen/Dense>

namespace kalman {

template <typename M, int N>
concept ProcessModel = requires(const M m, const Eigen::Vector<double, N>& x, double dt) {
    { m.predict(x, dt) } -> std::convertible_to<Eigen::Vector<double, N>>;
};

}  // namespace kalman
