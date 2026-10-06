# Tutorial 3: Attitude estimation on SO(3)

Orientation can't be filtered as a plain vector. Quaternions have a unit-norm constraint, and adding two rotations isn't a rotation. The **error-state** filter keeps the orientation on its manifold and puts the Gaussian on a small 3-D error around it.

This tutorial estimates orientation and gyro bias from:

- **a gyro at 100 Hz** that drives the prediction, biased by up to 0.02 rad/s,
- **an accelerometer and a magnetometer at 10 Hz**, which observe the known gravity and magnetic-field directions in the body frame.

The filter starts about **30° off** with no idea of the bias. The complete program is [`examples/tutorial_attitude.cpp`](../../examples/tutorial_attitude.cpp) (`ctest -R tutorial_attitude`).

## Describing the state

A manifold state is a class template with a tangent dimension and two operators:
- `boxplus` (⊞) applies a small error.
- `boxminus` (⊟) measures the difference between two states.

kalman::SO3 provides these for orientations, using right perturbations (`q ⊞ δ = q ⊗ Exp(δ)`). Combine it with the bias block by block:

```cpp
template <typename T>
struct AttitudeBias {
    static constexpr int DoF = 6;               // 3 for rotation + 3 for bias

    kalman::SO3<T> R;
    Eigen::Matrix<T, 3, 1> bias = Eigen::Matrix<T, 3, 1>::Zero();

    AttitudeBias boxplus(const Eigen::Matrix<T, 6, 1>& d) const {
        return {R.boxplus(d.template head<3>()), bias + d.template tail<3>()};
    }
    Eigen::Matrix<T, 6, 1> boxminus(const AttitudeBias& other) const {
        Eigen::Matrix<T, 6, 1> out;
        out << R.boxminus(other.R), bias - other.bias;
        return out;
    }
    template <typename U>
    AttitudeBias<U> cast() const { return {R.template cast<U>(), bias.template cast<U>()}; }
};
```

The state is templated on the scalar `T` so the filter can run autodiff through it.

## The models

The gyro reading is a **control input**, carried by the process model for that step:

```cpp
struct Gyro {
    Eigen::Vector3d omega;
    template <typename T>
    AttitudeBias<T> predict(const AttitudeBias<T>& x, double dt) const {
        const Eigen::Matrix<T, 3, 1> w = (omega.template cast<T>() - x.bias) * dt;
        return {{(x.R.q * kalman::so3::exp(w)).normalized()}, x.bias};
    }
};
```

The accelerometer and magnetometer each observe a known world direction rotated into the body frame:

```cpp
struct Direction {
    Eigen::Vector3d world;
    template <typename T>
    Eigen::Matrix<T, 3, 1> measure(const AttitudeBias<T>& x) const {
        return x.R.q.conjugate() * world.template cast<T>();
    }
};
```

## Running the filter

```cpp
using ESKF = kalman::ErrorStateKalmanFilter<AttitudeBias>;
ESKF eskf(AttitudeBias<double>{}, P0);

eskf.predict(Gyro{gyro_reading}, dt, Q);                    // every gyro sample
eskf.update(Direction{up}, accel_reading, Ra);              // at 10 Hz
eskf.update(Direction{field}, mag_reading, Rm);
```

You never write a Jacobian. The filter differentiates through your models and `⊞`/`⊟` to get:
- the error-state transition F,
- the measurement Jacobian H,
- the reset Jacobian G that re-centers the covariance after each correction.

The library's tests compare all three with their analytic forms to 1e-12.

## Output

```
t = 10 s   attitude error   0.54 deg   bias error 0.0011 rad/s
t = 20 s   attitude error   0.31 deg   bias error 0.0004 rad/s
...
t = 60 s   attitude error   0.32 deg   bias error 0.0002 rad/s
```

From a 30° start, the attitude is within a degree after 10 s, and the gyro bias is estimated to 0.0002 rad/s.
