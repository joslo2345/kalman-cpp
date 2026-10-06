# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

Done so far:
- Steps 1–3: the repo scaffold, the CMake build, and install/export rules.
- Step 5: in-house forward-mode autodiff in `autodiff.hpp`, using Ceres-style `Jet<T, N>` dual numbers registered with Eigen through `NumTraits` and `ScalarBinaryOpTraits`. It adds no external dependency.
- Step 6.1: `LinearFilter<N, M, Scalar>`.
- Step 6.2: `ExtendedKalmanFilter<N, Scalar>`.
- Step 6.3: `UnscentedKalmanFilter<N, Scalar>` and `SquareRootUnscentedKalmanFilter<N, Scalar>`.
  - Both use scaled Van der Merwe sigma points, set through `UnscentedParams`; the defaults are alpha=1, beta=2, kappa=0, which keep the weights non-negative.
  - The SR-UKF propagates the lower Cholesky factor with QR plus rank-1 `cholupdate`.
  - Q may be positive semi-definite, since `psd_sqrt` uses an eigendecomposition. R must be positive-definite.
  - The shared code is in `detail/unscented.hpp`.
- Step 6.4: `RtsSmoother<N, Scalar>`, one smoother for all four filters.
  - Every filter's `predict()` fills a `Prediction` record, read through `last_prediction()`. It holds the predicted mean and covariance, the cross-covariance `Cov(x_{k-1}, x_k)`, and a sequence counter.
  - The cross-covariance is `P F^T` for the KF and EKF, and comes from the sigma points for the UKF and SR-UKF, which gives the unscented RTS smoother.
  - Usage is `smoother.record(filter)` once per time step, then `smooth()`. Steps without a measurement are fine.
  - The sequence counter makes `record()` throw unless exactly one `predict()` ran in between.
  - Any new filter must fill `Prediction` in `predict()` to work with the smoother.
- Step 6.5: `AsyncFusion<Filter, Process>` in `fusion.hpp`.
  - It wraps an EKF, UKF or SR-UKF with a process model and a `Q(dt)` function. The linear KF is excluded because its F is fixed and can't vary with dt; the EKF with linear models gives the same results.
  - Each `add(t, model, z, R)` may use a different sensor model and size. A type-erased update is stored per measurement.
  - Late measurements are handled exactly by rewind and replay. The engine keeps a filter snapshot after every buffered measurement and drops entries older than `horizon`. A late measurement is inserted in time order and the later ones are replayed on copies, so a failure leaves the engine unchanged.
  - Equal timestamps are applied in arrival order.
  - Possible results: `applied`, `reordered`, `too_old`, `update_failed`.
  - `predicted(t)` returns a predicted copy and never changes the engine.
- Step 6.6: `ErrorStateKalmanFilter<S, Scalar>` in `error_state.hpp`, plus `so3.hpp`.
  - It is generic over a manifold state class template `S<T>` that provides `DoF`, `boxplus`, `boxminus` and `cast<U>()`; see the `ManifoldState` concept.
  - `kalman::SO3<T>` is a ready-made orientation state with right perturbation (`q ⊞ δ = q ⊗ Exp(δ)`).
  - Product states, such as `scenarios::AttitudeBias` in the tests, combine their parts block by block.
  - All three Jacobians come from autodiff through the model and `⊞`/`⊟`: process F, measurement H, and the error-reset G.
  - Models and states must therefore be templated on the scalar; there is no analytic-Jacobian path.
  - `so3::exp` and `so3::log` switch to Taylor expansions below about 1e-5 rad, so jet derivatives stay exact at the expansion point δ=0.
  - The ESKF doesn't fill `Prediction`, so the smoother doesn't support it. `AsyncFusion` doesn't fit it well either, since the gyro is a per-step control input carried by the process model.
- Square-root KF: `SquareRootLinearFilter<N, M, Scalar>` in `sqrt_linear_filter.hpp`. It propagates the Cholesky factor with the QR "array" algorithm and is the only filter that survives S4 in float32. It is about 3.5× slower than the Joseph-form KF on small problems.
- `detail::SmallCholesky`: an unrolled fixed-size Cholesky used by `joseph_update`. On 1×1 and 2×2 innovation covariances, Eigen's blocked `LLT` was the dominant per-step cost (S1 went from 72 to 38 ns). It keeps the positive-definiteness check that `S.inverse()` would lose.
- Step 7 (consistency): `kalman::diagnostics` in `diagnostics.hpp`.
  - `nees`/`nis` helpers, plus exact `chi2_cdf`/`chi2_quantile` from the regularized incomplete gamma function. They are accurate up to dof 5e4 and beyond.
  - `average_bounds(dof, runs, confidence)` gives the interval for a Monte Carlo mean.
  - `ConsistencyCheck` accumulates per-step NEES/NIS across runs and reports `fraction_inside()`.

The KF and EKF share `detail::joseph_update`, which uses an LLT solve and returns `false` without touching the state when S isn't positive-definite. Steps 5–9 are complete. The repo is public at https://github.com/joslo2345/kalman-cpp, and CI is green on Linux GCC 14, Linux Clang 18, macOS AppleClang and Windows MSVC.
- **Warnings:** they are errors only on the Clang jobs.
- **MSVC quirk:** MSVC can't deduce template sizes through `Eigen::Matrix<Scalar, K, L>` parameters when K = 1. That's a row vector, whose default storage option depends on K and L. Read the sizes from the argument types instead (see `detail::sqrt_covariance`).
- **Comparison job:** it uses Ubuntu's OpenCV 4.6 plus contrib. Frozen-vector results match macOS exactly.
- **Speed regression gate:** the guide's ">10% slower than the last release" check isn't implemented, because there is no release baseline yet.

The next step is Step 10: documentation.

OpenCV 5.0 (Homebrew, including the contrib modules) is installed. The `comparison_tests` executable builds only when CMake finds OpenCV.

Step 8 (benchmarks), in `bench/`:
- **Frozen scenarios:** S1–S5 live in `tests/vectors/` as `.npy` files with `SHA256SUMS`, generated by `scripts/make_vectors.py`. They are read by `tests/scenarios/vectors.hpp`. Never regenerate them after results exist.
- **Baselines:**
  - **mherb/kalman:** a git submodule at `bench/baselines/kalman`, pinned at `9f40c2f`. Its last upstream commit was in 2018.
  - **OpenCV EKF:** built on `cv::KalmanFilter` with a pseudo-measurement.
  - **OpenCV UKF:** from contrib `cv::detail::tracking`.
  - **Angle handling:** mherb/kalman and OpenCV's UKF have no angle hook. Their runners use a reference-angle workaround: before each update, `h(x)` returns bearings unwrapped near the predicted bearing. Without it their UKFs diverge on S3, which would be an unfair "failure".
- **Allocation counting:** heaptrack is Linux-only, so `bench_alloc` links `alloc_counter.dylib`, which interposes malloc and friends through DYLD_INTERPOSE. It self-checks that allocations inside OpenCV are counted (29 at setup), so the measured 0 per step for OpenCV is real.

Design decisions that differ from the guide's sketches:
- **EKF signature:** it is `ExtendedKalmanFilter<N>` rather than `<N, Mz>`.
  - Models are passed per call, as `predict(model, dt, Q)` and `update(model, z, R)`.
  - Mz is deduced from `z`, so one filter can fuse sensors of different sizes.
- **Model contracts** (C++20 concepts in `concepts.hpp`):
  - Process models provide `predict(x, dt)`; measurement models provide `measure(x)`.
  - A model may supply `jacobian(...)`, and the filter prefers it. Otherwise the model must be templated on its scalar type so the filter can autodiff it.
  - Models call math functions unqualified (`using std::sin; sin(x)`), so argument-dependent lookup finds the Jet overloads.
  - An optional `residual(z, z_pred)` handles angle wrapping. The unscented filters also use it to average sigma points around the central one, so bearings near ±π average correctly.
  - The UKF and SR-UKF only need `double` models, not templated ones.
  - The autodiff concepts must check the output scalar type exactly (`HasShapeAndScalar`). Eigen's converting constructors make a shape-only check accept double-only models.

Testing notes for the unscented filters:
- `scenarios::range_bearing()` by default is only mildly nonlinear, so the EKF and UKF are tied there.
- Use `scenarios::close_pass()` when a test needs the UKF to beat the EKF. Its UKF/EKF RMSE ratio was 0.75–0.86 over five disjoint blocks of 200 seeds.
- With alpha=1e-3 the central weight is about −1e6, which costs about 6 digits of precision. Tolerances against the exact KF are therefore 1e-5 there and 1e-10 for alpha=1.

The strongest smoother test compares against the exact batch solution of the whole linear-Gaussian problem (inverting the information matrix), including a gap of missing measurements. It needs a full-rank Q, because the constant-velocity Q from `cv_process_noise` is singular.

Testing notes for the error-state filter:
- Convergence and NEES are insensitive to the reset Jacobian G, which is close to the identity for small corrections.
- G is pinned down only by the analytic single-update test in `test_error_state.cpp`. It uses a correction of about 0.3 rad and compares against `blockdiag(J_r(δθ), I)`.
- The ESKF NEES bound is DoF ± 40%, because means over 20-seed blocks ranged from 5.1 to 6.9.

Notes on the comparison tests:
- **OpenCV-based EKF baseline:** `run_opencv_ekf` re-linearizes `measurementMatrix` each step and passes the pseudo-measurement `z − h(x̂) + H·x̂`, so OpenCV's linear residual equals the EKF innovation. A test checks that it agrees with our EKF before any claim is made against it.
- **Stability (S4):** prior variance 1e6, measurement variance 1e-6, float32. Every covariance-form filter loses information at the first predict, because 1e-6 added to 1e4 rounds away and the prior becomes singular to working precision.
  - OpenCV's short form collapses P to 0 at step 0.
  - Our Joseph-form KF fails at step 1. An earlier pass with Eigen `LLT` was rounding luck; its step-1 posterior was equally wrong.
  - Only `SquareRootLinearFilter` gets the right posterior. The stability test requires it to stay SPD and only reports the others.
- **Fair SPD check:** `is_spd` there uses a symmetry tolerance relative to the matrix's scale, so ordinary float round-off isn't counted as a failure.
- **Catch2 gotcha:** `UNSCOPED_INFO` only prints with `-s` and only when an assertion follows it. The comparison tests use `WARN("[report] ...")` instead, which prints on passing tests without `-s`, because `-s` would print 2M stability assertions. CI greps these lines into `comparison.md`.
- **Linker warnings:** with the macOS 26.5 SDK workaround, linking OpenCV warns that the dylibs were built for macOS 27. The warnings are harmless.

`tests/scenarios/` holds seeded generators that use `std::normal_distribution`. Its output differs between standard libraries, so use the frozen data in `tests/vectors/` for cross-library numbers. Every file in `tests/compile_fail/` must also be listed in the `foreach` in `tests/CMakeLists.txt`. Check that each one fails for the intended reason, not an unrelated error. `kalman-cpp-repo-guide.md` is the source of truth for the planned API, tests, and benchmarks. Read the relevant step there before adding a component.

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

Checks that CI runs (`.github/workflows/ci.yml`), all runnable locally. The linters are pinned in `.venv`, at the same versions as CI:
```bash
git ls-files '*.hpp' '*.cpp' | grep -v '^bench/baselines/' | xargs .venv/bin/clang-format --dry-run --Werror
ls tests/unit/*.cpp | xargs -P 6 -n 1 .venv/bin/clang-tidy -p build --quiet    # needs build/compile_commands.json
cmake -B build-asan -DKALMAN_SANITIZE=ON -DKALMAN_WARNINGS_AS_ERRORS=ON ... && ctest --test-dir build-asan -j6
cmake -B build-cov -DKALMAN_COVERAGE=ON ...   # then llvm-profdata merge + llvm-cov report on include/kalman
```
`kalman_configure_target()` in the top-level CMakeLists applies to our test and benchmark targets:
- warnings: `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`, or `/W4` on MSVC
- optional `-Werror`, ASan+UBSan and coverage
- Eigen as a system include

Benchmarks: one command builds `build-bench` (Release, `-DKALMAN_BUILD_BENCH=ON`). It runs timing, accuracy and allocation counting, then rewrites `results/results.csv` and the README table between the BENCH markers. On this Mac, pass the SDK workaround through `CMAKE_ARGS`:
```bash
CMAKE_ARGS="-G Ninja -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk" ./scripts/run_benchmarks.sh
./build-bench/bench/bench --benchmark_filter='S2'   # quick timing of one scenario
.venv/bin/python scripts/make_vectors.py --check   # verify the frozen scenarios
```
Python tooling uses the project-local `.venv/` (numpy, `scripts/requirements.txt`).

README notes:
- **Layout:** the README follows the CubeSandbox README's layout (centered header, badge rows, highlight-card tables).
- **Charts:** `scripts/make_charts.py` (run by `run_benchmarks.sh`) generates the charts from `results/results.csv` into `docs/assets/`, in light and dark variants served through `<picture>`.
- **Benchmark numbers by hand:** the hand-written summary table, the badges and the prose figures (tests count, coverage, speedups) don't update automatically. Edit them when the results change. Only the table between the BENCH markers is regenerated.
- **Snippets:** README code snippets come from `examples/*.cpp`, which are built and run as ctest tests. Keep the two in sync.

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
