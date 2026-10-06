// Timing: median time per predict+update step. Only the step loop is timed;
// filter setup and data conversion are excluded.
#include <benchmark/benchmark.h>

#include "runners.hpp"

namespace {

template <int N, int M>
void bench_ours(benchmark::State& state, const scenarios::Scenario<N, M>& sc) {
    for (auto _ : state) {
        state.PauseTiming();
        auto kf = bench::make_ours(sc);
        state.ResumeTiming();
        for (const auto& z : sc.zs) {
            kf.predict();
            kf.update(z);
        }
        benchmark::DoNotOptimize(kf.state().data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(sc.zs.size()));
}

template <int N, int M>
void bench_ours_sqrt(benchmark::State& state, const scenarios::Scenario<N, M>& sc) {
    for (auto _ : state) {
        state.PauseTiming();
        auto kf = bench::make_ours_sqrt(sc);
        state.ResumeTiming();
        for (const auto& z : sc.zs) {
            kf.predict();
            kf.update(z);
        }
        benchmark::DoNotOptimize(kf.state().data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(sc.zs.size()));
}

template <int N, int M>
void bench_opencv(benchmark::State& state, const scenarios::Scenario<N, M>& sc) {
    const std::vector<cv::Mat> zs = bench::to_cv(sc.zs);
    for (auto _ : state) {
        state.PauseTiming();
        cv::KalmanFilter kf = bench::make_opencv(sc);
        state.ResumeTiming();
        for (const auto& z : zs) {
            kf.predict();
            kf.correct(z);
        }
        benchmark::DoNotOptimize(kf.statePost.data);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(zs.size()));
}

template <int N, int M>
void bench_mherb(benchmark::State& state, const scenarios::Scenario<N, M>& sc) {
    std::vector<bench::MState<M, double>> zs(sc.zs.begin(), sc.zs.end());
    for (auto _ : state) {
        state.PauseTiming();
        bench::MherbLinear<N, M, double> kf(sc);
        state.ResumeTiming();
        for (const auto& z : zs) kf.step(z);
        benchmark::DoNotOptimize(kf.kf.getState().data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(zs.size()));
}

}  // namespace

int main(int argc, char** argv) {
    static const auto s1 = scenarios::load<2, 1>("S1");
    static const auto s2 = scenarios::load<4, 2>("S2");
    static const auto s5 = scenarios::load<15, 6>("S5");

    // Names are "library|scenario|filter|precision" so the converter can split them.
    benchmark::RegisterBenchmark("kalman-cpp|S1|KF|float64", [](benchmark::State& st) { bench_ours(st, s1); });
    benchmark::RegisterBenchmark("kalman-cpp-sqrt|S1|KF|float64",
                                 [](benchmark::State& st) { bench_ours_sqrt(st, s1); });
    benchmark::RegisterBenchmark("opencv|S1|KF|float64", [](benchmark::State& st) { bench_opencv(st, s1); });
    benchmark::RegisterBenchmark("mherb-kalman|S1|KF|float64", [](benchmark::State& st) { bench_mherb(st, s1); });
    benchmark::RegisterBenchmark("kalman-cpp|S2|KF|float64", [](benchmark::State& st) { bench_ours(st, s2); });
    benchmark::RegisterBenchmark("kalman-cpp-sqrt|S2|KF|float64",
                                 [](benchmark::State& st) { bench_ours_sqrt(st, s2); });
    benchmark::RegisterBenchmark("opencv|S2|KF|float64", [](benchmark::State& st) { bench_opencv(st, s2); });
    benchmark::RegisterBenchmark("mherb-kalman|S2|KF|float64", [](benchmark::State& st) { bench_mherb(st, s2); });
    benchmark::RegisterBenchmark("kalman-cpp|S5|KF|float64", [](benchmark::State& st) { bench_ours(st, s5); });
    benchmark::RegisterBenchmark("kalman-cpp-sqrt|S5|KF|float64",
                                 [](benchmark::State& st) { bench_ours_sqrt(st, s5); });
    benchmark::RegisterBenchmark("opencv|S5|KF|float64", [](benchmark::State& st) { bench_opencv(st, s5); });
    benchmark::RegisterBenchmark("mherb-kalman|S5|KF|float64", [](benchmark::State& st) { bench_mherb(st, s5); });

    benchmark::AddCustomContext("opencv_version", CV_VERSION);
    benchmark::AddCustomContext("mherb_kalman_commit", KALMAN_MHERB_COMMIT);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
