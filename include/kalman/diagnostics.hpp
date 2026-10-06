#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

namespace kalman::diagnostics {

// Normalized estimation error squared, e^T P^-1 e, where e = true - estimate.
// For a consistent filter it is chi-square distributed with N degrees of freedom.
template <int N, typename Scalar>
Scalar nees(const Eigen::Matrix<Scalar, N, 1>& error, const Eigen::Matrix<Scalar, N, N>& P) {
    const Eigen::LLT<Eigen::Matrix<Scalar, N, N>> llt(P);
    if (llt.info() != Eigen::Success) throw std::invalid_argument("nees: covariance is not positive-definite");
    return error.dot(llt.solve(error));
}

// Normalized innovation squared, y^T S^-1 y. Same distribution, with the
// measurement dimension as degrees of freedom. (Every filter also reports the
// NIS of its last update through nis().)
template <int M, typename Scalar>
Scalar nis(const Eigen::Matrix<Scalar, M, 1>& innovation, const Eigen::Matrix<Scalar, M, M>& S) {
    return nees(innovation, S);
}

// Regularized lower incomplete gamma function P(a, x) (Numerical Recipes 6.2):
// a series for x < a + 1, a continued fraction otherwise.
inline double regularized_gamma_p(double a, double x) {
    if (a <= 0.0 || x < 0.0) throw std::invalid_argument("regularized_gamma_p: need a > 0 and x >= 0");
    if (x == 0.0) return 0.0;
    const double log_prefix = a * std::log(x) - x - std::lgamma(a);
    constexpr double eps = 1e-15;
    constexpr int max_iter = 100000;

    if (x < a + 1.0) {
        double term = 1.0 / a, sum = term;
        for (int n = 1; n < max_iter && std::abs(term) > std::abs(sum) * eps; ++n) {
            term *= x / (a + n);
            sum += term;
        }
        return sum * std::exp(log_prefix);
    }

    // Lentz's method for the continued fraction of Q(a, x) = 1 - P(a, x).
    constexpr double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < max_iter; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < eps) break;
    }
    return 1.0 - std::exp(log_prefix) * h;
}

inline double chi2_cdf(double x, double dof) { return x <= 0.0 ? 0.0 : regularized_gamma_p(dof / 2.0, x / 2.0); }

// Inverse chi-square CDF, by bisection on the CDF.
inline double chi2_quantile(double p, double dof) {
    if (!(p > 0.0 && p < 1.0)) throw std::invalid_argument("chi2_quantile: p must be in (0, 1)");
    if (!(dof > 0.0)) throw std::invalid_argument("chi2_quantile: dof must be positive");
    double lo = 0.0, hi = dof + 10.0 * std::sqrt(2.0 * dof) + 10.0;
    while (chi2_cdf(hi, dof) < p) hi *= 2.0;
    for (int i = 0; i < 200 && hi - lo > 1e-14 * hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        (chi2_cdf(mid, dof) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

struct Bounds {
    double lower;
    double upper;
    bool contains(double value) const { return value >= lower && value <= upper; }
};

// Two-sided interval for the mean of `samples` independent chi2(dof) values:
// the sum is chi2(dof * samples), so the mean lies within
// [q(a/2), q(1 - a/2)] / samples with probability `confidence`.
inline Bounds average_bounds(int dof, int samples, double confidence = 0.95) {
    if (dof <= 0 || samples <= 0) throw std::invalid_argument("average_bounds: dof and samples must be positive");
    const double alpha = 1.0 - confidence;
    const double total = static_cast<double>(dof) * samples;
    return {chi2_quantile(alpha / 2, total) / samples, chi2_quantile(1 - alpha / 2, total) / samples};
}

// Monte Carlo consistency check (Bar-Shalom et al.): run the same filter on
// `runs` independent realizations, add each run's NEES (or NIS) at every time
// step, then compare the per-step average with the chi-square interval. For a
// consistent filter about `confidence` of the steps fall inside.
class ConsistencyCheck {
public:
    ConsistencyCheck(int dof, int runs, std::size_t steps) : dof_(dof), runs_(runs), sums_(steps, 0.0) {}

    void add(std::size_t step, double value) { sums_.at(step) += value; }

    double average(std::size_t step) const { return sums_.at(step) / runs_; }

    // Mean over all steps of the per-step averages; about dof when consistent.
    double overall_average() const {
        double total = 0.0;
        for (double s : sums_) total += s;
        return total / (static_cast<double>(runs_) * sums_.size());
    }

    // Fraction of steps whose average lies inside the interval.
    double fraction_inside(double confidence = 0.95) const {
        const Bounds b = average_bounds(dof_, runs_, confidence);
        std::size_t inside = 0;
        for (std::size_t k = 0; k < sums_.size(); ++k) inside += b.contains(average(k));
        return static_cast<double>(inside) / sums_.size();
    }

private:
    int dof_;
    int runs_;
    std::vector<double> sums_;
};

}  // namespace kalman::diagnostics
