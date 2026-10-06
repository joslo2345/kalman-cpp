# kalman-cpp

Header-only C++20 Kalman filters built on Eigen's fixed-size matrices:
- compile-time checked dimensions
- automatic Jacobians through forward-mode autodiff
- numerically robust updates (Joseph form and square-root filters)

## Start here

- [Tutorial 1: a constant-velocity tracker](tutorials/01-constant-velocity-tracker.md). A linear filter, RTS smoothing and a consistency check.
- [Tutorial 2: GPS + IMU fusion](tutorials/02-gps-imu-fusion.md). Multi-rate sensors and late measurements with kalman::AsyncFusion.
- [Tutorial 3: attitude estimation](tutorials/03-attitude-estimation.md). An error-state filter on SO(3) with gyro, accelerometer and magnetometer.

## Which filter?

| You have | Use |
|---|---|
| A linear model | kalman::LinearFilter |
| A linear model with a huge dynamic range, or float32 | kalman::SquareRootLinearFilter |
| A mildly nonlinear model | kalman::ExtendedKalmanFilter, with autodiff or your own Jacobians |
| A strongly nonlinear model | kalman::UnscentedKalmanFilter or kalman::SquareRootUnscentedKalmanFilter |
| A state on a manifold (orientation) | kalman::ErrorStateKalmanFilter with kalman::SO3 or your own state |
| A recorded run to post-process | kalman::RtsSmoother |
| Several sensors at different rates, arriving late | kalman::AsyncFusion |
| Doubts about your tuning | kalman::diagnostics::ConsistencyCheck |

## Model contracts

The filters are written against C++20 concepts (see concepts.hpp):

- **Process model:** `predict(x, dt)` returns the next state.
- **Measurement model:** `measure(x)` returns the expected measurement. It may also define `residual(z, z_pred)` for wrapped quantities such as angles.
- **Jacobians:** the EKF uses `jacobian(...)` when the model provides one. Otherwise the model must be templated on its scalar type, and the Jacobian is computed by autodiff. Call math functions unqualified (`using std::sin; sin(x)`).
- **Unscented filters:** they only need `double` models.

Every filter's `update()` returns `false`, leaving the estimate untouched, if the innovation covariance is not positive-definite.
