#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace kalman {

enum class FusionResult {
    applied,        // newest measurement, applied on top of the current estimate
    reordered,      // out-of-sequence: inserted in time order and later ones replayed
    too_old,        // older than the buffering horizon; ignored
    update_failed,  // the filter rejected it (e.g. innovation covariance not PD); ignored
};

/// Fuses asynchronous, multi-rate measurements into one model-based filter
/// (ExtendedKalmanFilter, UnscentedKalmanFilter or SquareRootUnscentedKalmanFilter).
///
/// Each add() may use a different measurement model and size. Measurements are
/// processed in timestamp order. One that arrives late (but within `horizon`
/// seconds of the newest) is handled exactly: the engine rewinds to a snapshot
/// taken just before its timestamp, inserts it, and replays the later ones, so
/// the result equals in-order processing. Measurements with equal timestamps
/// are applied in arrival order.
///
/// Measurements already in the buffer that fail during a replay stay buffered
/// but have no effect.
template <typename Filter, typename Process>
class AsyncFusion {
public:
    using State = typename Filter::State;
    using Cov = typename Filter::Cov;
    using Scalar = typename State::Scalar;
    using NoiseModel = std::function<Cov(double dt)>;

    static_assert(
        requires(Filter& f, const Process& p, double dt, const Cov& Q) { f.predict(p, dt, Q); },
        "AsyncFusion needs a filter with predict(process_model, dt, Q)");

    AsyncFusion(const Filter& filter, double t0, Process process, NoiseModel process_noise, double horizon)
        : process_(std::move(process)),
          noise_(std::move(process_noise)),
          horizon_(horizon),
          base_time_(t0),
          base_(filter),
          current_time_(t0),
          current_(filter) {
        if (!(horizon >= 0.0)) throw std::invalid_argument("AsyncFusion: horizon must be non-negative");
        if (!noise_) throw std::invalid_argument("AsyncFusion: process noise model is required");
    }

    template <typename Model, int Mz>
    FusionResult add(double t, Model h, const Eigen::Matrix<Scalar, Mz, 1>& z, const Eigen::Matrix<Scalar, Mz, Mz>& R) {
        if (t < base_time_) return FusionResult::too_old;

        Entry entry{t, [h = std::move(h), z, R](Filter& f) { return f.update(h, z, R); }, base_};

        // Insert after every buffered entry with time <= t (arrival order on ties).
        std::size_t i = log_.size();
        while (i > 0 && log_[i - 1].time > t) --i;

        Filter f = i == 0 ? base_ : log_[i - 1].after;
        double time = i == 0 ? base_time_ : log_[i - 1].time;
        if (!advance(f, time, t) || !entry.apply(f)) return FusionResult::update_failed;
        entry.after = f;
        time = t;

        // Replay later entries on copies, so nothing changes unless all of it succeeds.
        std::vector<Filter> replayed;
        replayed.reserve(log_.size() - i);
        for (std::size_t j = i; j < log_.size(); ++j) {
            if (!advance(f, time, log_[j].time)) return FusionResult::update_failed;
            log_[j].apply(f);
            replayed.push_back(f);
            time = log_[j].time;
        }

        const bool newest = i == log_.size();
        for (std::size_t j = i; j < log_.size(); ++j) log_[j].after = std::move(replayed[j - i]);
        log_.insert(log_.begin() + static_cast<std::ptrdiff_t>(i), std::move(entry));
        current_ = f;
        current_time_ = time;
        prune();
        return newest ? FusionResult::applied : FusionResult::reordered;
    }

    /// Time of the newest processed measurement (or t0).
    double time() const { return current_time_; }

    /// The filter at time(), after every buffered measurement.
    const Filter& filter() const { return current_; }
    const State& state() const { return current_.state(); }
    decltype(auto) covariance() const { return current_.covariance(); }

    /// A copy of the filter predicted forward to t >= time(); the engine itself is
    /// unchanged. Throws std::invalid_argument for t < time(), or
    /// std::runtime_error if the prediction fails.
    Filter predicted(double t) const {
        if (t < current_time_) throw std::invalid_argument("AsyncFusion::predicted: t is before time()");
        Filter f = current_;
        if (!advance(f, current_time_, t)) throw std::runtime_error("AsyncFusion::predicted: prediction failed");
        return f;
    }

    /// Number of measurements still buffered for out-of-sequence handling.
    std::size_t buffered() const { return log_.size(); }

private:
    struct Entry {
        double time;
        std::function<bool(Filter&)> apply;
        Filter after;  // filter state right after this measurement
    };

    bool advance(Filter& f, double from, double to) const {
        const double dt = to - from;
        if (dt <= 0.0) return true;
        if constexpr (std::is_same_v<decltype(f.predict(process_, dt, noise_(dt))), bool>) {
            return f.predict(process_, dt, noise_(dt));
        } else {
            f.predict(process_, dt, noise_(dt));
            return true;
        }
    }

    /// Drops entries that are too old to be rewound to; the newest dropped one
    /// becomes the new base snapshot.
    void prune() {
        while (!log_.empty() && log_.front().time < current_time_ - horizon_) {
            base_time_ = log_.front().time;
            base_ = std::move(log_.front().after);
            log_.pop_front();
        }
    }

    Process process_;
    NoiseModel noise_;
    double horizon_;

    double base_time_;  // measurements before this are too old
    Filter base_;       // filter state at base_time_
    std::deque<Entry> log_;

    double current_time_;
    Filter current_;
};

}  // namespace kalman
