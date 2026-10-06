# kalman_tracker (ROS 2 example)

A ROS 2 node that tracks a point from timestamped position measurements with kalman-cpp. It uses an EKF with a 3-D constant-velocity model, run through `kalman::AsyncFusion`.

Measurements are fused by **header stamp, not arrival order**. A message that arrives late, within `horizon` seconds, is inserted at its timestamp, and the later ones are replayed.

| Topic | Type | |
|---|---|---|
| `measurement` | `geometry_msgs/msg/PointStamped` | subscribed |
| `estimate` | `nav_msgs/msg/Odometry` | published: position, velocity, and their covariances |

| Parameter | Default | Meaning |
|---|---|---|
| `accel_sigma` | 0.5 | unmodeled acceleration (m/s²) |
| `measurement_sigma` | 1.0 | measurement noise (m) |
| `horizon` | 1.0 | how late a measurement may arrive (s) |

## Build

kalman-cpp is a plain CMake package, so install it first. Then build this package with colcon (tested on ROS 2 Jazzy in CI):

```bash
sudo apt install libeigen3-dev
cmake -S <kalman-cpp> -B build -DKALMAN_BUILD_TESTS=OFF -DKALMAN_BUILD_EXAMPLES=OFF -DKALMAN_FETCH_EIGEN=OFF
sudo cmake --install build

cp -r <kalman-cpp>/ros2/kalman_tracker ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select kalman_tracker
```

## Run

```bash
source install/setup.bash
ros2 launch kalman_tracker tracker.launch.py      # or: ros2 run kalman_tracker tracker_node
ros2 topic pub --once /target/position geometry_msgs/msg/PointStamped \
  "{header: {stamp: {sec: 1}, frame_id: map}, point: {x: 1.0, y: 2.0, z: 0.0}}"
ros2 topic echo /target/estimate
```
