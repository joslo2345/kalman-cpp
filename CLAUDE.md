# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

Done so far:
- the guide's Steps 1–3: the git repo, the directory layout, the CMake build with install/export rules, and the Catch2 test target
- Step 6.1: `LinearFilter<N, M, Scalar>` with a Joseph-form update, an LLT solve instead of a matrix inverse, and NIS. `update()` returns `false` if the innovation covariance S isn't positive-definite.

`concepts.hpp` and `linear_filter.hpp` are real code; the other headers are still stubs. `tests/scenarios/` holds seeded generators that use `std::normal_distribution`. Its output differs between standard libraries, so use the frozen data in `tests/vectors/` for cross-library numbers. The next step is Step 5/6.2: autodiff plus the EKF. `kalman-cpp-repo-guide.md` is the source of truth for the planned API, tests, and benchmarks. Read the relevant step there before adding a component.

**Local toolchain quirk:** on this Mac, the Command Line Tools linker can't read the macOS 27 SDK (`tapi error: ... unknown architecture`), so CMake's compiler check fails. Until the tools are updated, configure with `-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.

## What this project is

kalman-cpp is a header-only C++20 Kalman filter library built on Eigen (fixed-size types). The library is meant to beat existing C++ options (mainly OpenCV's `cv::KalmanFilter` and mherb/kalman) on:
- compile-time dimension checking, using templates and C++20 concepts
- automatic Jacobians for the EKF, through optional autodiff, with user-supplied Jacobians as a fallback
- the full filter family: KF (Joseph-form update), EKF, UKF, square-root UKF, and RTS smoother
- asynchronous multi-rate fusion with out-of-sequence measurements
- error-state filtering on SO(3)

Implement in the order given in Step 6 of the guide: linear KF → EKF → UKF/SR-UKF → RTS smoother → fusion layer → SO(3) error-state.

This is one of four sister implementations (C, C++, Python, Rust). All four share a test-vectors repo, included as a Git submodule under `tests/vectors/`. It holds the frozen benchmark scenarios S1–S5. Don't modify scenario data after results have been collected.

## Build and test commands

```bash
cmake -B build -G Ninja && cmake --build build   # KALMAN_BUILD_TESTS=ON by default
ctest --test-dir build                     # all tests (Catch2 v3 via catch_discover_tests)
ctest --test-dir build -R "<test name>"    # single test by name regex
./build/tests/unit_tests "[unit]"          # single Catch2 tag directly
```

Benchmarks (planned, not wired up yet; Release, `-DKALMAN_BUILD_BENCH=ON`):
```bash
cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DKALMAN_BUILD_BENCH=ON && cmake --build build-bench
./build-bench/bench --benchmark_repetitions=30 --benchmark_report_aggregates_only=true \
    --benchmark_out=results/gbench.json --benchmark_out_format=json
python scripts/gbench_to_csv.py results/gbench.json "$ENV"
./build-bench/accuracy "$ENV" >> results/results.csv
python scripts/make_table.py results/results.csv kalman-cpp
```

## Architecture decisions

- **Library target:** `kalman` is a CMake `INTERFACE` target, aliased as `kalman::kalman`. It pulls in Eigen 3.4.0 through FetchContent and requires `cxx_std_20`. Headers live in `include/kalman/`, and `kalman.hpp` is the umbrella header. Include `install()`/`export()` rules so `find_package(kalman)` works.
- **Dependencies:** OpenCV is an optional test-only dependency (`find_package(OpenCV QUIET)`). The library and `unit_tests` must build without it. `comparison_tests` are added only when OpenCV is found.
- **Test layout:**
  - `tests/unit/` tests our code in isolation.
  - `tests/comparison/` compares against OpenCV and a naive textbook filter.
  - `tests/compile_fail/` holds code that must not compile. Each file is registered as an `EXCLUDE_FROM_ALL` target, and a `WILL_FAIL` ctest builds it.
  - `tests/scenarios/` holds seeded problem generators.
  - `tests/vectors/` holds the shared data.
- **Correctness bar:** On linear Gaussian problems the goal is to match OpenCV to within 1e-9, not to beat it. The advantages come from stability (covariance stays SPD over 1M `float` steps), compile-time safety, speed, and features.
- **Test conventions:**
  - Seed all stochastic tests, and average them over many seeds (for example 200 for range-bearing) so they aren't flaky.
  - Report comparative results through `UNSCOPED_INFO("[report] ...")` lines instead of asserting on baseline behavior.
- **Benchmark fairness rules (Step 8):**
  - Every library gets the same inputs and the same precision.
  - Only predict+update is timed.
  - Missing baseline features are reported as "n/a".
  - Every result row records the library version, CPU, OS, and toolchain.
- **Benchmark output:**
  - Results go to `results/results.csv` with the schema `library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date`.
  - Benchmark names use the format `library|scenario|filter|precision`.
  - The README table goes between the `<!-- BENCH:START -->` and `<!-- BENCH:END -->` markers.
- **Heap allocations:** These are counted with heaptrack at the malloc level, because OpenCV's `fastMalloc` bypasses `operator new`. Fixed-size Eigen paths should allocate zero times per step.

## Planned CI

CI builds on GCC, Clang, AppleClang, and MSVC. It runs ASan and UBSan, `clang-tidy`, and a `clang-format` check, and measures coverage with gcov/llvm-cov. CI fails if speed regresses more than 10% from the last release.
