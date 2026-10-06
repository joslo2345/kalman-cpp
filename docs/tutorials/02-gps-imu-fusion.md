# Tutorial 2: GPS + IMU fusion

Real systems have sensors running at different rates whose readings arrive late. Here a vehicle drives a circle at 5 m/s and two sensors observe it:

- **an accelerometer at 100 Hz**, low-noise and immediate,
- **a GPS at 1 Hz**, noisy (2 m) and delivered **0.3 s after it was measured**.

kalman::AsyncFusion handles both problems. The complete program is [`examples/tutorial_gps_imu.cpp`](../../examples/tutorial_gps_imu.cpp) (`ctest -R tutorial_gps_imu`).

## The model

The state is `x = [px, py, vx, vy, ax, ay]` with constant acceleration, driven by white jerk. The process model is templated on the scalar, so the EKF can differentiate it automatically. Here it's linear, but the same code works for nonlinear models.

```cpp
struct ConstantAcceleration {
    template <typename T>
    Eigen::Matrix<T, 6, 1> predict(const Eigen::Matrix<T, 6, 1>& x, double dt) const {
        Eigen::Matrix<T, 6, 1> out = x;
        for (int i = 0; i < 2; ++i) {
            out(i) += x(i + 2) * dt + x(i + 4) * (0.5 * dt * dt);
            out(i + 2) += x(i + 4) * dt;
        }
        return out;
    }
};
```

Each sensor is a measurement model with its own size. One filter accepts both, and each size is still checked at compile time:

```cpp
struct GpsPosition {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 6, 1>& x) const { return x.template head<2>(); }
};
struct Accelerometer {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 6, 1>& x) const { return x.template tail<2>(); }
};
```

## The fusion engine

`AsyncFusion` wraps a filter, the process model and a process-noise function `Q(dt)`. It keeps a history `horizon` seconds long:

```cpp
kalman::AsyncFusion<kalman::ExtendedKalmanFilter<6>, ConstantAcceleration> fusion(
    ekf, /*t0=*/0.0, ConstantAcceleration{}, [](double dt) { return jerk_noise(dt, 0.5); },
    /*horizon=*/1.0);
```

Feed readings **in the order they arrive**, each with the timestamp of **when it was measured**:

```cpp
const auto result = r.gps ? fusion.add(r.stamp, GpsPosition{}, r.z, R_gps)
                          : fusion.add(r.stamp, Accelerometer{}, r.z, R_accel);
```

When a GPS fix arrives 0.3 s late, 30 accelerometer readings have already been applied. The engine rewinds to a snapshot taken just before the GPS timestamp, inserts the fix, and replays the later readings. The result is exactly what in-order processing would have produced; the library's tests check this to 1e-12.

`add()` reports what happened:

| Result | Meaning |
|---|---|
| `applied` | the newest reading, applied on top |
| `reordered` | a late reading, inserted at its timestamp |
| `too_old` | older than the horizon; ignored |
| `update_failed` | the filter rejected it (for example a bad R); ignored |

To read the estimate at a time between readings without changing the engine, use `fusion.predicted(t)`.

## Output

```
59 of 6060 readings arrived out of order and were inserted at their timestamp
position RMSE: GPS alone 2.83 m, fused 1.57 m
```

The accelerometer fills in the motion between GPS fixes, which almost halves the position error. Every late fix is still used at the right time.

**Next:** [Tutorial 3: attitude estimation](03-attitude-estimation.md)
