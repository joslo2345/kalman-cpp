#include <iostream>
#include <kalman/kalman.hpp>

int main() {
    // 1-D constant velocity: state [position, velocity], measuring position.
    using KF = kalman::LinearFilter</*N=*/2, /*M=*/1>;
    const double dt = 0.1;

    KF::Transition F;
    F << 1, dt, 0, 1;
    KF::Observation H;
    H << 1, 0;
    const KF::Cov Q = 1e-3 * KF::Cov::Identity();
    const KF::MeasCov R = KF::MeasCov::Constant(0.25);

    KF kf(F, H, Q, R, KF::State(0.0, 0.0), 10.0 * KF::Cov::Identity());

    for (double z : {0.11, 0.19, 0.32, 0.38, 0.52}) {
        kf.predict();
        kf.update(KF::Measurement(z));  // a 2-element measurement would not compile
    }
    std::cout << "position " << kf.state()(0) << ", velocity " << kf.state()(1) << ", NIS " << kf.nis() << '\n';
}
