# kalman-cpp

A modern, header-only C++20 Kalman filter library built on Eigen.

> **Status:** early scaffolding. The filters aren't implemented yet. See `kalman-cpp-repo-guide.md` for the roadmap.

## Goals

- **Compile-time safety:** the compiler checks matrix dimensions through templates and concepts.
- **Automatic Jacobians:** write `f(x)` and `h(x)` once, and the EKF derives the Jacobians through autodiff.
- **A full filter family:** KF, EKF, UKF, square-root UKF, and an RTS smoother.
- **Real-world sensor fusion:** asynchronous multi-rate sensors, out-of-sequence measurements, and state constraints.
- **Manifold support:** error-state filtering for orientations on SO(3).

## Build and test

Requires CMake ≥ 3.20 and a C++20 compiler. Eigen 3.4 and Catch2 v3 are fetched automatically.

```bash
cmake -B build
cmake --build build
ctest --test-dir build
```

## Use it from another project

```cmake
find_package(kalman REQUIRED)
target_link_libraries(my_app PRIVATE kalman::kalman)
```

## Benchmarks

All the numbers below come from `scripts/run_benchmarks.sh`. It checks the frozen scenarios in `tests/vectors/`, builds in Release, and writes `results/results.csv` and this table. "Ours vs best other" compares the best `kalman-cpp*` variant with the best other library. Above 1.00x means ours is better.

- **Machine:** Apple M3 Pro, macOS 27.0.1, Apple clang 21.0.0, `-O2 -mcpu=native` for every library. This is a laptop, not a dedicated benchmark machine, so treat timings as ±10%.
- **Libraries:**
  - **`kalman-cpp`** is the Joseph-form KF.
  - **`kalman-cpp-sqrt`** is the square-root KF (`SquareRootLinearFilter`).
  - **`opencv`** is OpenCV 5.0.0 from Homebrew.
  - **`mherb-kalman`** is commit `9f40c2f`. Its last commit was in 2018.
  - **`naive`** is a textbook filter with `P = (I − KH)P`.
- **S3 baselines:**
  - **OpenCV EKF:** OpenCV has no EKF class. This one is built on `cv::KalmanFilter` by re-linearizing every step, the usual approach. A test checks that it agrees with our EKF.
  - **OpenCV UKF:** this is the contrib `tracking` module's UKF (`cv::detail::tracking` in OpenCV 5).
  - **Angle handling:** neither OpenCV nor mherb/kalman has a hook for angles. Their runners unwrap bearings around the predicted bearing, which is what a user would do. Without that, their UKFs diverge (RMSE about 60) when a bearing crosses ±π.
- **S4 `steps_to_failure`:** a value of 1,000,000 (1e+06) means the filter never failed. Every covariance-form filter, ours included, loses information at the first predict in float32. A 1e-6 variance added to 1e4 rounds away, so they fail within two steps. Only the square-root form survives.
- **Speed trade-off:** on tiny problems the Joseph-form KF is about 1.5× slower than mherb/kalman. It pays for the Joseph form, the positive-definiteness check and NIS on every step.

<!-- BENCH:START -->
| Scenario | Filter | Precision | Metric | kalman-cpp | kalman-cpp-sqrt | mherb-kalman | naive | opencv | Ours vs best other |
|---|---|---|---|---|---|---|---|---|---|
| S1 | KF | float64 | max_abs_diff (state) | n/a | n/a | 2.728e-12 | 2.728e-12 | 2.728e-12 | n/a |
| S1 | KF | float64 | nees (-) | 1.991 | n/a | 1.991 | 1.991 | 1.991 | – |
| S1 | KF | float64 | rmse (state) | 0.3874 | n/a | 0.3874 | 0.3874 | 0.3874 | 1.00x |
| S1 | KF | float64 | time_per_step (ns) | 35.5 | 131.8 | 23 | n/a | 2840 | 0.65x |
| S2 | KF | float64 | heap_allocations (count) | 0 | 0 | 0 | n/a | 0 | – |
| S2 | KF | float64 | max_abs_diff (state) | n/a | n/a | 9.095e-13 | 1.364e-12 | 1.364e-12 | n/a |
| S2 | KF | float64 | nees (-) | 4.035 | n/a | 4.035 | 4.035 | 4.035 | – |
| S2 | KF | float64 | rmse (state) | 0.569 | n/a | 0.569 | 0.569 | 0.569 | 1.00x |
| S2 | KF | float64 | time_per_step (ns) | 96.81 | 325.8 | 62.06 | n/a | 2815 | 0.64x |
| S3 | EKF | float64 | nees (-) | 17.9 | n/a | 17.9 | n/a | 17.9 | – |
| S3 | EKF | float64 | rmse (state) | 5.72 | n/a | 5.72 | n/a | 5.72 | 1.00x |
| S3 | UKF | float64 | nees (-) | 8.778 | n/a | 8.748 | n/a | 8.735 | – |
| S3 | UKF | float64 | rmse (state) | 5.355 | n/a | 5.346 | n/a | 5.29 | 0.99x |
| S4 | KF | float32 | steps_to_failure (steps) | 1 | 1e+06 | 0 | 0 | 0 | ∞ |
| S5 | KF | float64 | nees (-) | 18 | n/a | 18 | 18 | 18 | – |
| S5 | KF | float64 | rmse (state) | 1.05 | n/a | 1.05 | 1.05 | 1.05 | 1.00x |
| S5 | KF | float64 | time_per_step (ns) | 2830 | 5573 | 2619 | n/a | 5392 | 0.93x |
<!-- BENCH:END -->

## License

MIT
