# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/). Before 1.0, minor releases may change the API.

## [0.1.0] - 2026-10-06

First release.

### Added

- **Filters:**
  - `LinearFilter` (Joseph-form update)
  - `SquareRootLinearFilter` (QR array form)
  - `ExtendedKalmanFilter`
  - `UnscentedKalmanFilter`
  - `SquareRootUnscentedKalmanFilter`
  - `ErrorStateKalmanFilter`, for states on manifolds, with `SO3`
- **Automatic Jacobians:** forward-mode autodiff (`Jet<T, N>`) integrated with Eigen. The EKF and error-state filter differentiate models templated on the scalar, and use hand-written Jacobians when a model provides them.
- **Compile-time checks:** C++20 concepts for process, measurement and manifold models, with measurement sizes checked at compile time.
- **`RtsSmoother`:** works with every filter, including the unscented RTS smoother.
- **`AsyncFusion`:** multi-rate, multi-sensor fusion with exact handling of out-of-sequence measurements (rewind and replay within a time horizon).
- **`diagnostics`:** NEES/NIS, exact chi-square quantiles, and Monte Carlo consistency checks.
- **Robustness:** every update rejects a non-positive-definite innovation covariance without touching the state, and nothing is heap-allocated per step.
- **Benchmarks:**
  - five frozen scenarios (`tests/vectors/`, with SHA-256 checksums)
  - comparisons against OpenCV and mherb/kalman
  - one-command reproduction (`scripts/run_benchmarks.sh`)
- **Documentation:** an API reference built by Doxygen, three tested tutorials, and runnable examples.
- **Packaging:**
  - a CMake package (`find_package(kalman)`, target `kalman::kalman`)
  - a Conan recipe
  - a ROS 2 example package (`ros2/kalman_tracker`)
- **CI:** Linux GCC/Clang, macOS AppleClang, Windows MSVC, ASan+UBSan, clang-tidy, coverage, install, Conan, ROS 2 and OpenCV comparison jobs.

### Known limitations

- **Speed on small problems:** on 2–4-state linear filters the Joseph-form KF is about 1.5× slower than mherb/kalman's short-form update. That's the cost of the Joseph form, the positive-definiteness check and NIS on every step.
- **Covariance form in float32:** in single precision, covariance-form filters, including `LinearFilter`, cannot handle covariances spanning more than float's dynamic range (benchmark S4). Use `SquareRootLinearFilter` for those.
- **Error-state filter:** `ErrorStateKalmanFilter` works with neither `RtsSmoother` nor `AsyncFusion` yet.
- **Regression gate:** CI has no speed-regression gate yet. This release is the first baseline.

[0.1.0]: https://github.com/joslo2345/kalman-cpp/releases/tag/v0.1.0
