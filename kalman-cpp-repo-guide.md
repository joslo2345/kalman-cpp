# Building a Kalman Filter Library in C++

A step-by-step guide to setting up the repository for a modern, header-only C++ Kalman filter library.

## Project goals

The library should fill the gaps in the current C++ ecosystem:

- **Compile-time safety.** Matrix dimensions are checked by the compiler through templates.
- **Automatic Jacobians.** Users write the nonlinear model once, and the EKF derives Jacobians through autodiff.
- **A full filter family.** It offers KF, EKF, UKF, square-root variants, and an RTS smoother.
- **Real-world sensor fusion.** It supports asynchronous multi-rate sensors, out-of-sequence measurements, and state constraints.
- **Manifold support.** It provides error-state filtering for orientations (quaternions/SO(3)).

## Step 1: Create the repository

```bash
mkdir kalman-cpp && cd kalman-cpp
git init
```

Add a permissive `LICENSE` (MIT or Apache-2.0) and a `.gitignore` covering `build/`, `.cache/`, and IDE folders.

## Step 2: Set up the directory layout

```
kalman-cpp/
├── include/kalman/
│   ├── kalman.hpp            # umbrella header
│   ├── concepts.hpp          # C++20 concepts for models
│   ├── linear_filter.hpp
│   ├── ekf.hpp
│   ├── ukf.hpp
│   ├── sqrt_ukf.hpp
│   ├── smoother.hpp
│   ├── fusion.hpp            # async multi-sensor handling
│   └── diagnostics.hpp       # NIS / NEES
├── tests/
│   └── vectors/              # shared reference test data
├── examples/
├── bench/
├── docs/
├── cmake/
├── CMakeLists.txt
├── README.md
└── LICENSE
```

## Step 3: Configure the build system

Make the library a header-only `INTERFACE` target that depends on Eigen.

```cmake
cmake_minimum_required(VERSION 3.20)
project(kalman_cpp VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0)
FetchContent_MakeAvailable(Eigen3)

add_library(kalman INTERFACE)
add_library(kalman::kalman ALIAS kalman)
target_include_directories(kalman INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(kalman INTERFACE Eigen3::Eigen)
target_compile_features(kalman INTERFACE cxx_std_20)

option(KALMAN_BUILD_TESTS "Build tests" ON)
if(KALMAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

Add `install()` and `export()` rules so users can call `find_package(kalman)`.

## Step 4: Design the API with concepts

Use fixed-size Eigen types and C++20 concepts so mistakes are caught at compile time.

```cpp
template <typename M, int N>
concept ProcessModel = requires(const M m, const Eigen::Vector<double, N>& x, double dt) {
    { m.predict(x, dt) } -> std::convertible_to<Eigen::Vector<double, N>>;
};

template <int N, int Mz>
class ExtendedKalmanFilter {
public:
    using State = Eigen::Vector<double, N>;
    using Cov   = Eigen::Matrix<double, N, N>;

    template <ProcessModel<N> Model>
    void predict(const Model& model, double dt);

    template <typename MeasModel>
    void update(const MeasModel& h, const Eigen::Vector<double, Mz>& z,
                const Eigen::Matrix<double, Mz, Mz>& R);

    const State& state() const;
    const Cov& covariance() const;
    double nis() const;
};
```

## Step 5: Integrate automatic differentiation

Choose one autodiff approach and make it optional:

- Use the [autodiff](https://autodiff.github.io) library, which is header-only and works with Eigen.
- Use Ceres-style dual-number jets.
- Let users supply their own Jacobians as a fallback.

The goal is that users only write `f(x)` and `h(x)`, and Jacobians are generated automatically.

## Step 6: Implement in this order

1. Implement the linear KF with the Joseph-form update.
2. Add the EKF with autodiff Jacobians.
3. Add the UKF and a square-root UKF.
4. Add the RTS smoother.
5. Add the asynchronous fusion layer, which time-orders measurements, handles multi-rate sensors, and buffers for out-of-sequence data.
6. Add an error-state filter for orientation on SO(3).

## Step 7: Write unit tests that prove the library is better

Use [Catch2](https://github.com/catchorg/Catch2) v3, pulled in with FetchContent, and run everything through `ctest`.

### Define "better" before writing tests

For a linear system with Gaussian noise, the Kalman filter is the optimal estimator. Two correct implementations should produce the **same** answer, so on accuracy the goal is to *match* OpenCV, not beat it. Measurable improvements come from stability, safety, speed, and features:

| Criterion | What the test does | Pass condition |
|---|---|---|
| Equivalence | Runs the same problem through our filter and `cv::KalmanFilter` | States match within 1e-9 |
| Numerical stability | Runs 1,000,000 steps of an ill-conditioned problem in `float` | Our covariance stays symmetric and positive-definite; the step where OpenCV fails is recorded |
| Compile-time safety | Tries to compile code with mismatched dimensions | Compilation fails |
| Autodiff correctness | Compares automatic Jacobians to analytic ones | Match within 1e-12 |
| Nonlinear accuracy | Runs a strongly nonlinear range-bearing problem | Our UKF has lower RMSE than an EKF built on OpenCV's linear filter |
| Consistency | Runs Monte Carlo trials and computes NIS/NEES | Falls inside the 95% chi-squared bounds |
| Speed | Benchmarks predict+update at small fixed sizes | Faster than OpenCV, with no regressions over 10% |

### Make OpenCV an optional test dependency

Comparison tests should never be required to build the library itself.

```cmake
find_package(OpenCV QUIET COMPONENTS core video)

add_executable(unit_tests
    unit/test_linear.cpp unit/test_ekf.cpp unit/test_ukf.cpp unit/test_autodiff.cpp)
target_link_libraries(unit_tests PRIVATE kalman::kalman Catch2::Catch2WithMain)
catch_discover_tests(unit_tests)

if(OpenCV_FOUND)
    add_executable(comparison_tests
        comparison/test_equivalence.cpp comparison/test_stability.cpp
        comparison/test_nonlinear.cpp)
    target_link_libraries(comparison_tests PRIVATE
        kalman::kalman Catch2::Catch2WithMain ${OpenCV_LIBS})
    catch_discover_tests(comparison_tests)
endif()
```

### Organize the tests

```
tests/
├── unit/                 # our code in isolation
├── comparison/           # our code vs OpenCV and a naive textbook filter
├── compile_fail/         # code that must NOT compile
├── scenarios/            # shared problem generators with fixed seeds
└── vectors/              # shared reference test data
```

### Example: equivalence with OpenCV

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <opencv2/video/tracking.hpp>
#include <kalman/kalman.hpp>
#include "scenarios/constant_velocity.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("Linear KF matches OpenCV on 2D constant-velocity tracking", "[comparison]") {
    auto sc = scenarios::constant_velocity_2d(/*steps=*/1000, /*seed=*/42);

    kalman::LinearFilter<4, 2> ours(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);

    cv::KalmanFilter theirs(4, 2, 0, CV_64F);
    scenarios::configure_opencv(theirs, sc);

    for (const auto& z : sc.measurements) {
        ours.predict();
        ours.update(z);

        theirs.predict();
        cv::Mat est = theirs.correct(scenarios::to_cv(z));

        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(ours.state()(i), WithinAbs(est.at<double>(i), 1e-9));
        }
    }
}
```

### Example: long-run numerical stability

```cpp
#include <Eigen/Eigenvalues>

template <typename Mat>
bool is_spd(const Mat& P) {
    if ((P - P.transpose()).cwiseAbs().maxCoeff() > 1e-4f) return false;
    return Eigen::LLT<Mat>(P).info() == Eigen::Success;
}

TEST_CASE("Covariance stays SPD over 1M float steps", "[comparison][stability]") {
    auto sc = scenarios::ill_conditioned<float>(/*seed=*/7);

    kalman::LinearFilter<4, 2, float> ours(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    cv::KalmanFilter theirs(4, 2, 0, CV_32F);
    scenarios::configure_opencv(theirs, sc);

    long opencv_failed_at = -1;
    for (long k = 0; k < 1'000'000; ++k) {
        auto z = sc.measure(k);

        ours.predict();
        ours.update(z);
        REQUIRE(is_spd(ours.covariance()));

        if (opencv_failed_at < 0) {
            theirs.predict();
            theirs.correct(scenarios::to_cv(z));
            if (!is_spd(scenarios::from_cv<float, 4>(theirs.errorCovPost)))
                opencv_failed_at = k;
        }
    }
    UNSCOPED_INFO("[report] OpenCV lost SPD at step " << opencv_failed_at);
}
```

The test only *requires* our filter to stay stable. OpenCV's result is reported for the comparison table, since it isn't guaranteed to fail on every platform.

### Example: automatic Jacobians are correct

```cpp
TEST_CASE("Autodiff Jacobian matches analytic Jacobian", "[unit][autodiff]") {
    scenarios::RangeBearingModel model;
    for (const auto& x : scenarios::random_states<4>(/*count=*/100, /*seed=*/3)) {
        Eigen::Matrix<double, 2, 4> auto_J = kalman::jacobian(model, x);
        Eigen::Matrix<double, 2, 4> exact_J = model.analytic_jacobian(x);
        REQUIRE((auto_J - exact_J).cwiseAbs().maxCoeff() < 1e-12);
    }
}
```

### Example: nonlinear accuracy

```cpp
TEST_CASE("UKF beats an OpenCV-based EKF on range-bearing tracking", "[comparison]") {
    double ours_rmse = 0, opencv_ekf_rmse = 0;
    for (int seed = 0; seed < 200; ++seed) {
        auto sc = scenarios::range_bearing(/*steps=*/500, seed);
        ours_rmse      += scenarios::rmse(scenarios::run_ukf(sc), sc.truth);
        opencv_ekf_rmse += scenarios::rmse(scenarios::run_opencv_ekf(sc), sc.truth);
    }
    UNSCOPED_INFO("[report] UKF RMSE " << ours_rmse / 200
                  << " vs OpenCV EKF RMSE " << opencv_ekf_rmse / 200);
    REQUIRE(ours_rmse < opencv_ekf_rmse);
}
```

Averaging over 200 seeds keeps this test from being flaky. Use a scenario with strong nonlinearity, such as a target passing close to the sensor, so the difference is real.

### Example: dimension mismatches don't compile

Put intentionally broken code in `tests/compile_fail/dimension_mismatch.cpp`:

```cpp
#include <kalman/kalman.hpp>
int main() {
    kalman::LinearFilter<4, 2> kf(/* ... */);
    Eigen::Vector3d z;   // wrong size: filter expects 2 measurements
    kf.update(z);        // must be a compile error
}
```

Register it as a test that passes only when the build fails:

```cmake
add_executable(compile_fail_dims EXCLUDE_FROM_ALL compile_fail/dimension_mismatch.cpp)
target_link_libraries(compile_fail_dims PRIVATE kalman::kalman)
add_test(NAME compile_fail_dims
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target compile_fail_dims)
set_tests_properties(compile_fail_dims PROPERTIES WILL_FAIL TRUE)
```

OpenCV's `cv::Mat` checks sizes only at runtime, so this is a guarantee it can't offer.

### Consistency tests

Run 500 Monte Carlo trials where the true state is known. Check that average NIS and NEES fall inside the 95% chi-squared bounds, which confirms the filter is neither overconfident nor underconfident.

### Speed comparison

Catch2 includes micro-benchmarks, which are useful for quick local checks:

```cpp
TEST_CASE("predict+update speed vs OpenCV", "[!benchmark]") {
    auto sc = scenarios::constant_velocity_2d(1, 0);
    kalman::LinearFilter<4, 2> ours(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
    cv::KalmanFilter theirs(4, 2, 0, CV_64F);
    scenarios::configure_opencv(theirs, sc);
    auto z = sc.measurements[0];
    cv::Mat zm = scenarios::to_cv(z);

    BENCHMARK("kalman-cpp 4x2") { ours.predict(); ours.update(z); return ours.state()(0); };
    BENCHMARK("OpenCV 4x2")     { theirs.predict(); return theirs.correct(zm).at<double>(0); };
}
```

For regression gating in CI, use the Google Benchmark suite from Step 8, which gives more stable numbers.

### Publish a comparison report

Have CI collect the `[report]` lines and benchmark results into a `comparison.md` table and upload it as a build artifact. Fail the build if any of our pass conditions break, or if speed regresses by more than 10% from the last release. This report is the evidence behind any claim of being better than OpenCV or other existing libraries.

## Step 8: Produce comparison numbers against existing libraries

The unit tests in the previous step prove the library is *correct*. This step produces the *numbers* for the README: a table that shows, scenario by scenario, how this library compares to the existing ones. Everything runs from one command, writes raw data to a CSV file, and regenerates the table automatically, so anyone can reproduce the results.

### Benchmark scenarios

Use the same five scenarios in all four repos (C, C++, Python, Rust). Store them as data files (true states, measurements, F, H, Q, R, x0, P0) in the shared test-vectors repo and pull it in as a Git submodule. Because every library reads exactly the same inputs, the numbers are comparable against the external libraries *and* across our four implementations.

| ID | Scenario | State / measurement size | Filters | Length | What it shows |
|---|---|---|---|---|---|
| S1 | 1D constant velocity | 2 / 1 | KF | 10,000 steps | Overhead on tiny problems |
| S2 | 2D constant velocity | 4 / 2 | KF | 10,000 steps | A typical tracking workload |
| S3 | Range-bearing tracking | 4 / 2 | EKF, UKF | 500 steps × 200 seeds | Accuracy on a nonlinear problem |
| S4 | Ill-conditioned problem | 4 / 2 | KF | 1,000,000 steps, 32-bit floats | Numerical stability |
| S5 | INS error-state | 15 / 6 | KF / EKF | 10,000 steps | Speed on a larger state |

Freeze the scenario files before collecting results. Changing a scenario after seeing which library wins makes the numbers meaningless.

### Metrics

| Metric (CSV name) | Unit | How it's measured | Better is |
|---|---|---|---|
| `time_per_step` | ns | Median of 30 timed runs of predict+update, after a warm-up run | Lower |
| `rmse` | state units | Root-mean-square error against the true states in the scenario file | Lower |
| `max_abs_diff` | state units | Largest difference from each baseline's estimate (S1, S2) | Close to 0 |
| `nees` | – | Average normalized estimation error squared, compared with 95% chi-squared bounds | Inside bounds |
| `steps_to_failure` | steps | First step where the covariance is no longer symmetric positive-definite (S4) | Higher |
| `heap_allocations` | count | Calls to allocation functions during the step loop, measured with heaptrack | Lower |

### What to compare against

| Library | Scenarios | Why | How to include it |
|---|---|---|---|
| [OpenCV](https://opencv.org) `cv::KalmanFilter` | S1, S2, S4, S5 | The most widely used C++ Kalman filter | Install OpenCV 4.x from your package manager or vcpkg; record `CV_VERSION` |
| [mherb/kalman](https://github.com/mherb/kalman) | S1, S2, S3, S5 | A well-known Eigen-based, header-only library with EKF and UKF | Add as a Git submodule in `bench/baselines/kalman`, pinned to a commit |
| Naive textbook filter | S4 | Shows what an unstabilized hand-written filter does | Your own implementation from Step 7 |

OpenCV has only a linear filter, so it's "n/a" for S3. mherb/kalman has EKF and UKF classes. Write its runner following the example in its repository, and check its recent commit activity before relying on it.

### The harness

Use [Google Benchmark](https://github.com/google/benchmark) with the real OpenCV API. Filter setup is excluded from timing, and measurements are converted to `cv::Mat` before timing starts, so only predict+update is measured.

```cpp
// bench/bench_main.cpp
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>      // must come after Eigen
#include <opencv2/video/tracking.hpp>
#include <kalman/kalman.hpp>
#include "scenarios/scenario.hpp"

template <int N, int M>
void configure_opencv(cv::KalmanFilter& kf, const scenarios::Scenario<N, M>& sc) {
    cv::eigen2cv(sc.F,  kf.transitionMatrix);
    cv::eigen2cv(sc.H,  kf.measurementMatrix);
    cv::eigen2cv(sc.Q,  kf.processNoiseCov);
    cv::eigen2cv(sc.R,  kf.measurementNoiseCov);
    cv::eigen2cv(sc.P0, kf.errorCovPost);
    cv::eigen2cv(sc.x0, kf.statePost);
}

template <int N, int M>
void bench_ours(benchmark::State& state, const scenarios::Scenario<N, M>& sc) {
    for (auto _ : state) {
        state.PauseTiming();
        kalman::LinearFilter<N, M> kf(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0);
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
    std::vector<cv::Mat> zs;
    for (const auto& z : sc.zs) {
        cv::Mat m;
        cv::eigen2cv(z, m);
        zs.push_back(m);
    }
    for (auto _ : state) {
        state.PauseTiming();
        cv::KalmanFilter kf(N, M, 0, CV_64F);
        configure_opencv(kf, sc);
        state.ResumeTiming();
        for (const auto& z : zs) {
            kf.predict();
            kf.correct(z);
        }
        benchmark::DoNotOptimize(kf.statePost.data);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(zs.size()));
}

int main(int argc, char** argv) {
    static const auto s1 = scenarios::load<2, 1>("S1");
    static const auto s2 = scenarios::load<4, 2>("S2");
    static const auto s5 = scenarios::load<15, 6>("S5");

    // Names are "library|scenario|filter|precision" so the converter can split them.
    benchmark::RegisterBenchmark("kalman-cpp|S1|KF|float64", [](benchmark::State& st) { bench_ours(st, s1); });
    benchmark::RegisterBenchmark("opencv|S1|KF|float64",     [](benchmark::State& st) { bench_opencv(st, s1); });
    benchmark::RegisterBenchmark("kalman-cpp|S2|KF|float64", [](benchmark::State& st) { bench_ours(st, s2); });
    benchmark::RegisterBenchmark("opencv|S2|KF|float64",     [](benchmark::State& st) { bench_opencv(st, s2); });
    benchmark::RegisterBenchmark("kalman-cpp|S5|KF|float64", [](benchmark::State& st) { bench_ours(st, s5); });
    benchmark::RegisterBenchmark("opencv|S5|KF|float64",     [](benchmark::State& st) { bench_opencv(st, s5); });
    // Add the mherb/kalman runners the same way.

    benchmark::AddCustomContext("opencv_version", CV_VERSION);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
```

Build in Release mode with the same flags for everything (for example `-O2 -march=native`; state the flags next to the table). Run it with 30 repetitions and JSON output:

```bash
./build-bench/bench --benchmark_repetitions=30 \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=results/gbench.json --benchmark_out_format=json
```

Convert the median of each benchmark into the shared CSV format with `scripts/gbench_to_csv.py`:

```python
"""Usage: python scripts/gbench_to_csv.py results/gbench.json "commit,cpu,os,toolchain,date" """
import csv
import json
import sys

data = json.load(open(sys.argv[1]))
env = sys.argv[2].split(",")
versions = {"kalman-cpp": env[0],
            "opencv": data["context"].get("opencv_version", "unknown"),
            "mherb-kalman": "pinned-commit"}  # replace with the submodule's commit hash

with open("results/results.csv", "a", newline="") as f:
    out = csv.writer(f)
    for b in data["benchmarks"]:
        if b.get("aggregate_name") != "median":
            continue
        library, scenario, filt, precision = b["run_name"].split("/")[0].split("|")
        ns_per_step = 1e9 / b["items_per_second"]
        out.writerow([library, versions.get(library, "unknown"), scenario, filt, precision,
                      "time_per_step", f"{ns_per_step:.2f}", "ns", *env])
```

Accuracy (`rmse`, `nees`, `max_abs_diff`) and stability (`steps_to_failure`) aren't timing results, so compute them in a separate `bench/accuracy_main.cpp` that runs every library once per scenario (200 seeds for S3) and prints CSV rows directly.

### Counting heap allocations

OpenCV allocates through its own `cv::fastMalloc`, which bypasses a replaced `operator new`, so count allocations at the `malloc` level with [heaptrack](https://github.com/KDE/heaptrack). Build a small `bench_alloc` binary that runs one library on S2, then:

```bash
heaptrack ./build-bench/bench_alloc kalman-cpp S2
heaptrack ./build-bench/bench_alloc opencv S2
heaptrack_print heaptrack.bench_alloc.*.zst | grep "calls to allocation functions"
```

Subtract a run with zero steps for each library, so setup allocations don't count, and record the difference as `heap_allocations`. Fixed-size Eigen types should give zero for our library; `cv::Mat` temporaries usually allocate on every step.

### One command to produce everything

```bash
ENV="$(git rev-parse --short HEAD),$(uname -m),$(uname -s),gcc-$(gcc -dumpversion),$(date -I)"
cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DKALMAN_BUILD_BENCH=ON
cmake --build build-bench
./build-bench/bench --benchmark_repetitions=30 --benchmark_report_aggregates_only=true \
    --benchmark_out=results/gbench.json --benchmark_out_format=json
python scripts/gbench_to_csv.py results/gbench.json "$ENV"
./build-bench/accuracy "$ENV" >> results/results.csv
python scripts/make_table.py results/results.csv kalman-cpp
```

### Rules for a fair comparison

- Every library gets the same scenario file, random seed, initial state, and noise matrices.
- Every library uses the same floating-point precision. If a baseline doesn't support the precision a scenario needs, its cell is marked "n/a".
- Only predict+update is timed. Setup, file loading, data conversion, and printing are excluded.
- Every result records the library version (or commit hash), CPU, OS, and toolchain.
- A feature a baseline doesn't have (for example a UKF) is reported as "n/a", never as a failure.
- Publish the raw CSV and the scripts, so the authors of the other libraries can check the setup and reproduce the numbers.

### Output format

Every benchmark run appends rows to `results/results.csv` with this schema:

```
library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date
```

The script below turns the CSV into the README table and calculates how our library compares with the best other library for each row (above 1.00x means ours is better). Save it as `scripts/make_table.py`. It's Python in every repo, since it's only tooling.

```python
"""Usage: python scripts/make_table.py results/results.csv <our-library-name>"""
import csv
import sys
from collections import defaultdict

LOWER_IS_BETTER = {"time_per_step", "cycles_per_step", "rmse", "heap_allocations",
                   "peak_memory", "flash_bytes", "ram_bytes"}
HIGHER_IS_BETTER = {"steps_to_failure"}

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
ours = sys.argv[2]
libs = [ours] + sorted({r["library"] for r in rows} - {ours})

cells = defaultdict(dict)
for r in rows:
    key = (r["scenario"], r["filter"], r["precision"], r["metric"], r["unit"])
    cells[key][r["library"]] = float(r["value"])


def ratio(metric, vals):
    # Libraries named "<ours>-something" are our own variants (e.g. another backend)
    others = [v for lib, v in vals.items() if not lib.startswith(ours)]
    if ours not in vals or not others:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        best = min(others)
        return "–" if vals[ours] == 0 else f"{best / vals[ours]:.2f}x"
    if metric in HIGHER_IS_BETTER:
        best = max(others)
        return "–" if best == 0 else f"{vals[ours] / best:.2f}x"
    return "–"  # nees, max_abs_diff: read the values directly


print("| Scenario | Filter | Precision | Metric | " + " | ".join(libs) + " | Ours vs best other |")
print("|" + "---|" * (len(libs) + 5))
for (scen, filt, prec, metric, unit), vals in sorted(cells.items()):
    values = [f"{vals[lib]:.4g}" if lib in vals else "n/a" for lib in libs]
    print(f"| {scen} | {filt} | {prec} | {metric} ({unit}) | " + " | ".join(values)
          + f" | {ratio(metric, vals)} |")
```

Paste the output into the README between `<!-- BENCH:START -->` and `<!-- BENCH:END -->` markers, or have CI do it.

### Where to run the numbers

GitHub-hosted CI runners share hardware, so their timings can vary by 10–20% between runs. Use them to catch regressions, but produce the published numbers on one dedicated machine, with nothing else running and a fixed CPU frequency (on Linux, set the `performance` governor). State that machine's specs above the table.

### What the README table will look like

The values below are placeholders until the benchmarks run.

| Scenario | Filter | Precision | Metric | kalman-cpp | mherb-kalman | naive | opencv | Ours vs best other |
|---|---|---|---|---|---|---|---|---|
| S1 | KF | float64 | time_per_step (ns) | – | – | n/a | – | – |
| S2 | KF | float64 | time_per_step (ns) | – | – | n/a | – | – |
| S2 | KF | float64 | heap_allocations (count) | – | – | n/a | – | – |
| S3 | EKF | float64 | rmse (state) | – | – | n/a | n/a | – |
| S3 | UKF | float64 | rmse (state) | – | – | n/a | n/a | – |
| S4 | KF | float32 | steps_to_failure (steps) | – | n/a | – | – | – |
| S5 | KF | float64 | time_per_step (ns) | – | – | n/a | – | – |

## Step 9: Add continuous integration

Create a GitHub Actions workflow that:

- Builds on Linux (GCC, Clang), macOS (AppleClang), and Windows (MSVC).
- Runs tests with AddressSanitizer and UndefinedBehaviorSanitizer.
- Runs `clang-tidy` and checks `clang-format`.
- Measures coverage with `gcov`/`llvm-cov`.

## Step 10: Write documentation

- In the README, include a quick-start example, a feature comparison table, and benchmark numbers.
- Generate API docs with Doxygen, optionally styled with a theme such as doxygen-awesome.
- Add tutorials for a constant-velocity tracker, GPS+IMU fusion, and attitude estimation.

## Step 11: Package and release

- Tag releases with semantic versioning.
- Submit ports to [vcpkg](https://github.com/microsoft/vcpkg) and [Conan Center](https://conan.io/center).
- Provide a ROS 2 example package to reach the robotics community.
