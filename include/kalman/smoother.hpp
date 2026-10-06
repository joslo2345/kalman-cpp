#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include "kalman/detail/joseph_update.hpp"
#include "kalman/prediction.hpp"

namespace kalman {

// Rauch-Tung-Striebel fixed-interval smoother.
//
// Works with every filter in the library: record the filter once per time step
// (after that step's updates, if any), then call smooth().
//
//   kalman::RtsSmoother<4> smoother;
//   smoother.record(kf);                        // initial state
//   for (const auto& z : zs) {
//       kf.predict();
//       kf.update(z);                           // optional: missing data is fine
//       smoother.record(kf);
//   }
//   auto smoothed = smoother.smooth();          // one estimate per record
//
// The backward pass is
//
//   C_k     = D_{k+1} P_{k+1|k}^-1
//   x_{k|T} = x_{k|k} + C_k (x_{k+1|T} - x_{k+1|k})
//   P_{k|T} = P_{k|k} + C_k (P_{k+1|T} - P_{k+1|k}) C_k^T
//
// where D_{k+1} = Cov(x_k, x_{k+1}) comes from the filter's prediction: P F^T
// for the KF and EKF, and the sigma-point cross-covariance for the unscented
// filters (which makes this the unscented RTS smoother).
template <int N, typename Scalar = double>
class RtsSmoother {
public:
    using State = Eigen::Matrix<Scalar, N, 1>;
    using Cov = Eigen::Matrix<Scalar, N, N>;

    struct Estimate {
        State x;
        Cov P;
    };

    // Records the filter's current estimate. Between two records the filter
    // must have run exactly one predict(); throws std::logic_error otherwise.
    template <typename Filter>
    void record(const Filter& filter) {
        record(filter.last_prediction(), filter.state(), filter.covariance());
    }

    // Lower-level form: the filtered estimate at this step and the prediction
    // that led to it. The prediction is ignored for the first record.
    void record(const Prediction<N, Scalar>& pred, const State& x, const Cov& P) {
        if (!steps_.empty() && pred.sequence != last_sequence_ + 1) {
            throw std::logic_error("RtsSmoother: exactly one predict() is required between records");
        }
        steps_.push_back({{x, P}, pred});
        last_sequence_ = pred.sequence;
    }

    // Smoothed estimates, one per record. Throws std::runtime_error if a
    // predicted covariance is not positive-definite.
    std::vector<Estimate> smooth() const {
        std::vector<Estimate> out(steps_.size());
        if (steps_.empty()) return out;

        out.back() = steps_.back().filtered;
        for (std::size_t k = steps_.size() - 1; k-- > 0;) {
            const Estimate& filt = steps_[k].filtered;
            const Prediction<N, Scalar>& next = steps_[k + 1].pred;

            const Eigen::LLT<Cov> llt(next.P);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("RtsSmoother: predicted covariance is not positive-definite");
            }
            // P_{k+1|k} is symmetric, so C^T = P_{k+1|k}^-1 D^T.
            const Cov C = llt.solve(next.cross.transpose()).transpose();

            out[k].x = filt.x + C * (out[k + 1].x - next.x);
            out[k].P = filt.P + C * (out[k + 1].P - next.P) * C.transpose();
            detail::symmetrize(out[k].P);
        }
        return out;
    }

    std::size_t size() const { return steps_.size(); }
    void clear() { steps_.clear(); }

private:
    struct Step {
        Estimate filtered;
        Prediction<N, Scalar> pred;
    };

    std::vector<Step> steps_;
    std::uint64_t last_sequence_ = 0;
};

}  // namespace kalman
