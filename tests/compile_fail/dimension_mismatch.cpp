// This file must NOT compile: the filter expects a 2-element measurement.
#include <kalman/kalman.hpp>

int main() {
    using KF = kalman::LinearFilter<4, 2>;
    KF kf(KF::Transition::Identity(), KF::Observation::Zero(), KF::Cov::Identity(), KF::MeasCov::Identity(),
          KF::State::Zero(), KF::Cov::Identity());
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    kf.update(z);
}
