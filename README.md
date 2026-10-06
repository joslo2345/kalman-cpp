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

<!-- BENCH:START -->
_Not yet run._
<!-- BENCH:END -->

## License

MIT
