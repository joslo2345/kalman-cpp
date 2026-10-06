// Accuracy and stability metrics, printed as rows of results/results.csv:
//   library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date
// Usage: accuracy "commit,cpu,os,toolchain,date"
#include <cstdio>
#include <string>

#include "runners.hpp"

namespace {

std::string env;  // "commit,cpu,os,toolchain,date"
std::string ours_version;

std::string version_of(const std::string& library) {
    if (library.rfind("kalman-cpp", 0) == 0) return ours_version;
    if (library == "opencv") return CV_VERSION;
    if (library == "mherb-kalman") return KALMAN_MHERB_COMMIT;
    return "-";
}

void row(const std::string& library, const std::string& scenario, const std::string& filter,
         const std::string& precision, const std::string& metric, double value, const std::string& unit) {
    std::printf("%s,%s,%s,%s,%s,%s,%.10g,%s,%s\n", library.c_str(), version_of(library).c_str(), scenario.c_str(),
                filter.c_str(), precision.c_str(), metric.c_str(), value, unit.c_str(), env.c_str());
}

template <int N>
double rmse(const std::vector<bench::Estimate<N>>& est, const std::vector<Eigen::Matrix<double, N, 1>>& truth) {
    double sum = 0.0;
    for (std::size_t k = 0; k < est.size(); ++k) sum += (est[k].x - truth[k]).squaredNorm();
    return std::sqrt(sum / est.size());
}

template <int N>
double mean_nees(const std::vector<bench::Estimate<N>>& est, const std::vector<Eigen::Matrix<double, N, 1>>& truth) {
    double sum = 0.0;
    for (std::size_t k = 0; k < est.size(); ++k) {
        sum += kalman::diagnostics::nees(Eigen::Matrix<double, N, 1>(truth[k] - est[k].x), est[k].P);
    }
    return sum / est.size();
}

template <int N>
double max_abs_diff(const std::vector<bench::Estimate<N>>& a, const std::vector<bench::Estimate<N>>& b) {
    double m = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) m = std::max(m, (a[k].x - b[k].x).cwiseAbs().maxCoeff());
    return m;
}

template <int N, int M>
void linear_scenario(const std::string& name, bool report_diff) {
    const auto sc = scenarios::load<N, M>(name);

    std::vector<bench::Estimate<N>> ours, opencv, mherb;
    auto kf = bench::make_ours(sc);
    for (const auto& z : sc.zs) {
        kf.predict();
        kf.update(z);
        ours.push_back({kf.state(), kf.covariance()});
    }
    cv::KalmanFilter cvkf = bench::make_opencv(sc);
    for (const auto& z : bench::to_cv(sc.zs)) {
        cvkf.predict();
        cvkf.correct(z);
        opencv.push_back({bench::from_cv<double, N>(cvkf.statePost), bench::from_cv<double, N, N>(cvkf.errorCovPost)});
    }
    bench::MherbLinear<N, M, double> mk(sc);
    for (const auto& z : sc.zs) {
        mk.step(bench::MState<M, double>(z));
        mherb.push_back({mk.kf.getState(), mk.kf.getCovariance()});
    }

    std::vector<bench::Estimate<N>> naive;
    bench::NaiveFilter<N, M, double> nf(sc);
    for (const auto& z : sc.zs) {
        nf.step(z);
        naive.push_back({nf.x, nf.P});
    }

    for (const auto& [lib, est] :
         {std::pair{"kalman-cpp", &ours}, {"opencv", &opencv}, {"mherb-kalman", &mherb}, {"naive", &naive}}) {
        row(lib, name, "KF", "float64", "rmse", rmse(*est, sc.truth), "state");
        row(lib, name, "KF", "float64", "nees", mean_nees(*est, sc.truth), "-");
    }
    if (report_diff) {
        row("opencv", name, "KF", "float64", "max_abs_diff", max_abs_diff(ours, opencv), "state");
        row("mherb-kalman", name, "KF", "float64", "max_abs_diff", max_abs_diff(ours, mherb), "state");
        row("naive", name, "KF", "float64", "max_abs_diff", max_abs_diff(ours, naive), "state");
    }
}

void range_bearing() {
    const auto sc = scenarios::load_range_bearing("S3");
    struct Run {
        const char* library;
        const char* filter;
        double sq = 0.0, nees = 0.0;
    };
    Run runs[6] = {{"kalman-cpp", "EKF"},   {"kalman-cpp", "UKF"}, {"mherb-kalman", "EKF"},
                   {"mherb-kalman", "UKF"}, {"opencv", "EKF"},     {"opencv", "UKF"}};
    std::size_t count = 0;
    for (std::size_t s = 0; s < sc.zs.size(); ++s) {
        kalman::ExtendedKalmanFilter<4> ekf(sc.x0, sc.P0);
        kalman::UnscentedKalmanFilter<4> ukf(sc.x0, sc.P0);
        std::vector<bench::Estimate<4>> est[6];
        for (const auto& z : sc.zs[s]) {
            ekf.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q);
            ekf.update(scenarios::RangeBearingModel{}, z, sc.R);
            est[0].push_back({ekf.state(), ekf.covariance()});
            ukf.predict(scenarios::ConstantVelocity2D{}, sc.dt, sc.Q);
            ukf.update(scenarios::RangeBearingModel{}, z, sc.R);
            est[1].push_back({ukf.state(), ukf.covariance()});
        }
        est[2] = bench::run_mherb_range_bearing<Kalman::ExtendedKalmanFilter<bench::MState4>>(sc, s);
        est[3] = bench::run_mherb_range_bearing<Kalman::UnscentedKalmanFilter<bench::MState4>>(sc, s);
        est[4] = bench::run_opencv_ekf_range_bearing(sc, s);
        est[5] = bench::run_opencv_ukf_range_bearing(sc, s);
        for (int r = 0; r < 6; ++r) {
            const double e = rmse(est[r], sc.truth[s]);
            runs[r].sq += e * e * est[r].size();
            runs[r].nees += mean_nees(est[r], sc.truth[s]) * est[r].size();
        }
        count += sc.zs[s].size();
    }
    for (const auto& r : runs) {
        row(r.library, "S3", r.filter, "float64", "rmse", std::sqrt(r.sq / count), "state");
        row(r.library, "S3", r.filter, "float64", "nees", r.nees / count, "-");
    }
}

// First step whose posterior covariance is not symmetric positive-definite;
// the scenario length if it never fails.
template <typename Step, typename Cov>
long steps_to_failure(std::size_t steps, Step step, Cov cov) {
    for (std::size_t k = 0; k < steps; ++k) {
        step(k);
        if (!bench::is_spd(cov())) return static_cast<long>(k);
    }
    return static_cast<long>(steps);
}

void stability() {
    const auto sc = scenarios::load<4, 2, float>("S4");
    const std::size_t n = sc.zs.size();

    auto kf = bench::make_ours(sc);
    row("kalman-cpp", "S4", "KF", "float32", "steps_to_failure",
        steps_to_failure(
            n,
            [&](std::size_t k) {
                kf.predict();
                kf.update(sc.zs[k]);
            },
            [&] { return kf.covariance(); }),
        "steps");

    auto sr = bench::make_ours_sqrt(sc);
    row("kalman-cpp-sqrt", "S4", "KF", "float32", "steps_to_failure",
        steps_to_failure(
            n,
            [&](std::size_t k) {
                sr.predict();
                sr.update(sc.zs[k]);
            },
            [&] { return sr.covariance(); }),
        "steps");

    cv::KalmanFilter cvkf = bench::make_opencv(sc);
    const auto zs = bench::to_cv(sc.zs);
    row("opencv", "S4", "KF", "float32", "steps_to_failure",
        steps_to_failure(
            n,
            [&](std::size_t k) {
                cvkf.predict();
                cvkf.correct(zs[k]);
            },
            [&] { return bench::from_cv<float, 4, 4>(cvkf.errorCovPost); }),
        "steps");

    bench::MherbLinear<4, 2, float> mk(sc);
    row("mherb-kalman", "S4", "KF", "float32", "steps_to_failure",
        steps_to_failure(
            n, [&](std::size_t k) { mk.step(bench::MState<2, float>(sc.zs[k])); },
            [&] { return Eigen::Matrix4f(mk.kf.getCovariance()); }),
        "steps");

    bench::NaiveFilter<4, 2, float> naive(sc);
    row("naive", "S4", "KF", "float32", "steps_to_failure",
        steps_to_failure(n, [&](std::size_t k) { naive.step(sc.zs[k]); }, [&] { return naive.P; }), "steps");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s \"commit,cpu,os,toolchain,date\"\n", argv[0]);
        return 2;
    }
    env = argv[1];
    ours_version = env.substr(0, env.find(','));

    linear_scenario<2, 1>("S1", true);
    linear_scenario<4, 2>("S2", true);
    linear_scenario<15, 6>("S5", false);
    range_bearing();
    stability();
}
