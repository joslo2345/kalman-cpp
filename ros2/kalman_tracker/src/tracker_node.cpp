// Tracks a point from timestamped position measurements with kalman-cpp.
//
// Subscribes:  measurement (geometry_msgs/PointStamped)
// Publishes:   estimate    (nav_msgs/Odometry: position, velocity, covariances)
//
// Measurements are fused by header stamp, not arrival order: a message that
// arrives late (within `horizon` seconds) is inserted at its timestamp and the
// later ones are replayed, exactly as if everything had arrived in order.
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <kalman/kalman.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace {

/// State [px, py, pz, vx, vy, vz] moving at constant velocity.
struct ConstantVelocity3D {
    template <typename T>
    Eigen::Matrix<T, 6, 1> predict(const Eigen::Matrix<T, 6, 1>& x, double dt) const {
        Eigen::Matrix<T, 6, 1> out = x;
        out.template head<3>() += x.template tail<3>() * dt;
        return out;
    }
};

/// Measures position.
struct Position3D {
    template <typename T>
    Eigen::Matrix<T, 3, 1> measure(const Eigen::Matrix<T, 6, 1>& x) const {
        return x.template head<3>();
    }
};

/// Discrete white-noise acceleration, per axis q * [dt^4/4, dt^3/2; dt^3/2, dt^2].
Eigen::Matrix<double, 6, 6> process_noise(double dt, double accel_sigma) {
    const double q = accel_sigma * accel_sigma;
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    for (int axis = 0; axis < 3; ++axis) {
        Q(axis, axis) = q * dt * dt * dt * dt / 4;
        Q(axis, axis + 3) = Q(axis + 3, axis) = q * dt * dt * dt / 2;
        Q(axis + 3, axis + 3) = q * dt * dt;
    }
    return Q;
}

using Filter = kalman::ExtendedKalmanFilter<6>;
using Fusion = kalman::AsyncFusion<Filter, ConstantVelocity3D>;

class TrackerNode : public rclcpp::Node {
public:
    TrackerNode() : Node("kalman_tracker") {
        accel_sigma_ = declare_parameter("accel_sigma", 0.5);       // m/s^2, unmodeled acceleration
        meas_sigma_ = declare_parameter("measurement_sigma", 1.0);  // m
        horizon_ = declare_parameter("horizon", 1.0);               // s, how late a message may arrive
        R_ = meas_sigma_ * meas_sigma_ * Eigen::Matrix3d::Identity();

        publisher_ = create_publisher<nav_msgs::msg::Odometry>("estimate", 10);
        subscription_ = create_subscription<geometry_msgs::msg::PointStamped>(
            "measurement", 10, [this](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) { on_measurement(*msg); });
        RCLCPP_INFO(get_logger(), "kalman-cpp %s tracker ready", kalman::version);
    }

private:
    void on_measurement(const geometry_msgs::msg::PointStamped& msg) {
        const rclcpp::Time stamp(msg.header.stamp);
        const Eigen::Vector3d z(msg.point.x, msg.point.y, msg.point.z);

        if (!fusion_) {  // start at the first measurement, with unknown velocity
            t0_ = stamp;
            frame_id_ = msg.header.frame_id;
            Eigen::Matrix<double, 6, 1> x0;
            x0 << z, Eigen::Vector3d::Zero();
            Eigen::Matrix<double, 6, 1> p0;
            p0 << R_.diagonal(), Eigen::Vector3d::Constant(100.0);
            const double accel_sigma = accel_sigma_;
            fusion_.emplace(
                Filter(x0, p0.asDiagonal()), 0.0, ConstantVelocity3D{},
                [accel_sigma](double dt) { return process_noise(dt, accel_sigma); }, horizon_);
            publish();
            return;
        }

        // Seconds since the first measurement keep full double precision.
        switch (fusion_->add((stamp - t0_).seconds(), Position3D{}, z, R_)) {
            case kalman::FusionResult::applied:
            case kalman::FusionResult::reordered:
                publish();
                break;
            case kalman::FusionResult::too_old:
                RCLCPP_WARN(get_logger(), "dropped a measurement older than the %.2f s horizon", horizon_);
                break;
            case kalman::FusionResult::update_failed:
                RCLCPP_WARN(get_logger(), "the filter rejected a measurement");
                break;
        }
    }

    void publish() {
        const auto& x = fusion_->state();
        const auto P = fusion_->covariance();

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = t0_ + rclcpp::Duration::from_seconds(fusion_->time());
        odom.header.frame_id = frame_id_;
        odom.pose.pose.position.x = x(0);
        odom.pose.pose.position.y = x(1);
        odom.pose.pose.position.z = x(2);
        odom.pose.pose.orientation.w = 1.0;  // not estimated
        odom.twist.twist.linear.x = x(3);
        odom.twist.twist.linear.y = x(4);
        odom.twist.twist.linear.z = x(5);

        // 6x6 row-major covariances over (x, y, z, roll, pitch, yaw). The
        // orientation and angular velocity are not estimated: huge variance.
        odom.pose.covariance.fill(0.0);
        odom.twist.covariance.fill(0.0);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                odom.pose.covariance[static_cast<std::size_t>(r * 6 + c)] = P(r, c);
                odom.twist.covariance[static_cast<std::size_t>(r * 6 + c)] = P(r + 3, c + 3);
            }
            odom.pose.covariance[static_cast<std::size_t>((r + 3) * 7)] = 1e9;
            odom.twist.covariance[static_cast<std::size_t>((r + 3) * 7)] = 1e9;
        }
        publisher_->publish(odom);
    }

    double accel_sigma_, meas_sigma_, horizon_;
    Eigen::Matrix3d R_;
    rclcpp::Time t0_;
    std::string frame_id_;
    std::optional<Fusion> fusion_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr subscription_;
};

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackerNode>());
    rclcpp::shutdown();
    return 0;
}
