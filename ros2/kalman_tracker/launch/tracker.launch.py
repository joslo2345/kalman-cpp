from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="kalman_tracker",
            executable="tracker_node",
            name="kalman_tracker",
            parameters=[{"accel_sigma": 0.5, "measurement_sigma": 1.0, "horizon": 1.0}],
            remappings=[("measurement", "/target/position"), ("estimate", "/target/estimate")],
        ),
    ])
