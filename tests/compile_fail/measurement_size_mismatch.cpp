// This file must NOT compile: the model predicts 2 values but z has 3.
#include <kalman/kalman.hpp>

struct TwoOutputs {
    template <typename T>
    Eigen::Matrix<T, 2, 1> measure(const Eigen::Matrix<T, 4, 1>& x) const {
        return x.template head<2>();
    }
};

int main() {
    kalman::ExtendedKalmanFilter<4> ekf(Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity());
    ekf.update(TwoOutputs{}, Eigen::Vector3d::Zero().eval(), Eigen::Matrix3d::Identity().eval());
}
