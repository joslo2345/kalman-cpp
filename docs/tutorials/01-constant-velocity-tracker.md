# Tutorial 1: A constant-velocity tracker

This tutorial tracks a target moving in 2-D from noisy position measurements. It shows how to:

1. set up a linear Kalman filter with dimensions checked at compile time,
2. improve past estimates with the RTS smoother,
3. check that the filter is tuned correctly with the normalized innovation squared (NIS).

The complete program is [`examples/tutorial_cv_tracker.cpp`](../../examples/tutorial_cv_tracker.cpp). It is built and run as a test (`ctest -R tutorial_cv_tracker`).

## The model

The state is position and velocity, `x = [px, py, vx, vy]`. Between samples (every `dt = 0.1` s) the target keeps its velocity:

```
px' = px + vx·dt        vx' = vx
py' = py + vy·dt        vy' = vy
```

Real targets accelerate a little. That unmodeled acceleration enters as **process noise** Q. A sensor measures position with 1 m standard deviation, which gives the **measurement noise** R.

```cpp
using KF = kalman::LinearFilter<4, 2>;  // 4 states, 2 measurements

KF::Transition F = KF::Transition::Identity();
F(0, 2) = F(1, 3) = dt;

KF::Observation H = KF::Observation::Zero();
H(0, 0) = H(1, 1) = 1.0;               // we observe px and py

// Discrete white-noise acceleration: per axis q * [dt^4/4, dt^3/2; dt^3/2, dt^2]
KF::Cov Q = ...;
const KF::MeasCov R = meas_sigma * meas_sigma * KF::MeasCov::Identity();

KF kf(F, H, Q, R, x0, P0);
```

The sizes are part of the type. `kf.update()` only accepts a 2-element measurement, and passing a 3-element vector is a compile error, not a runtime crash.

## Filtering

Each step is a `predict()` followed by an `update()`:

```cpp
kf.predict();
if (!kf.update(z)) { /* innovation covariance not positive-definite: z was rejected */ }
```

`update()` uses the Joseph form, which keeps the covariance symmetric and positive-definite under round-off. It returns `false` instead of corrupting the estimate when a measurement can't be used.

## Smoothing

A filter only uses measurements up to the current time. When you post-process a recorded run, the **RTS smoother** revisits every step using the measurements that came after it. Record the filter once per step and call `smooth()` at the end:

```cpp
kalman::RtsSmoother<4> smoother;
smoother.record(kf);              // the initial state
for (...) {
    kf.predict();
    kf.update(z);
    smoother.record(kf);          // exactly one predict() between records
}
const auto smoothed = smoother.smooth();   // smoothed[k].x, smoothed[k].P
```

## Is the filter tuned correctly?

If Q and R match reality, the innovation `y = z − H·x` has the covariance the filter predicts. In that case the NIS `yᵀ S⁻¹ y` averages to the measurement dimension (here 2). `kf.nis()` reports it after every update, and `diagnostics::average_bounds` gives the interval the average should fall in:

```cpp
const auto bounds = kalman::diagnostics::average_bounds(/*dof=*/2, /*samples=*/steps);
```

A mean NIS above the interval means the filter is **overconfident**: Q or R is too small. Below it means the filter is **underconfident**.

## Output

```
position RMSE: measurements 1.00 m, filtered 0.40 m, smoothed 0.22 m
mean NIS 1.88 (95% interval for a consistent filter: [1.78, 2.23])
```

Filtering cuts the position error by 60%, and smoothing halves it again.

**Next:** [Tutorial 2: GPS + IMU fusion](02-gps-imu-fusion.md)
