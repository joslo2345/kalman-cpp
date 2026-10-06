// Heap allocations during the predict+update loop on S2, one CSV row per
// library. Setup is excluded by resetting the counter after it.
// Usage: bench_alloc "commit,cpu,os,toolchain,date"
#include <cstdio>
#include <string>

#include "runners.hpp"

extern "C" void alloc_counter_reset();
extern "C" unsigned long alloc_counter_get();

namespace {

template <typename Setup, typename Run>
unsigned long count_allocations(Setup setup, Run run) {
    auto state = setup();
    alloc_counter_reset();
    run(state);
    return alloc_counter_get();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s \"commit,cpu,os,toolchain,date\"\n", argv[0]);
        return 2;
    }
    const std::string env = argv[1];
    const std::string ours_version = env.substr(0, env.find(','));
    const auto sc = scenarios::load<4, 2>("S2");
    const auto zs_cv = bench::to_cv(sc.zs);
    const std::vector<bench::MState<2, double>> zs_mherb(sc.zs.begin(), sc.zs.end());

    // Sanity check that interposition is active.
    alloc_counter_reset();
    void* volatile probe = std::malloc(16);
    std::free(probe);
    if (alloc_counter_get() == 0) {
        std::fprintf(stderr, "bench_alloc: allocation counter is not active\n");
        return 1;
    }
    // ...and that it also sees allocations made inside OpenCV's library
    // (constructing a filter allocates its matrices), so a 0 below is real.
    alloc_counter_reset();
    { const cv::KalmanFilter setup_probe = bench::make_opencv(sc); }
    const unsigned long opencv_setup = alloc_counter_get();
    if (opencv_setup == 0) {
        std::fprintf(stderr, "bench_alloc: allocations inside OpenCV are not being counted\n");
        return 1;
    }
    std::fprintf(stderr, "bench_alloc: OpenCV filter setup made %lu allocations (counter OK)\n", opencv_setup);

    const unsigned long ours = count_allocations([&] { return bench::make_ours(sc); },
                                                 [&](auto& kf) {
                                                     for (const auto& z : sc.zs) {
                                                         kf.predict();
                                                         kf.update(z);
                                                     }
                                                 });
    const unsigned long ours_sqrt = count_allocations([&] { return bench::make_ours_sqrt(sc); },
                                                      [&](auto& kf) {
                                                          for (const auto& z : sc.zs) {
                                                              kf.predict();
                                                              kf.update(z);
                                                          }
                                                      });
    const unsigned long opencv = count_allocations([&] { return bench::make_opencv(sc); },
                                                   [&](auto& kf) {
                                                       for (const auto& z : zs_cv) {
                                                           kf.predict();
                                                           kf.correct(z);
                                                       }
                                                   });
    const unsigned long mherb = count_allocations(
        [&] { return std::make_unique<bench::MherbLinear<4, 2, double>>(sc); },
        [&](auto& kf) {
            for (const auto& z : zs_mherb) kf->step(z);
        });

    auto row = [&](const char* lib, const std::string& version, unsigned long value) {
        std::printf("%s,%s,S2,KF,float64,heap_allocations,%lu,count,%s\n", lib, version.c_str(), value, env.c_str());
    };
    row("kalman-cpp", ours_version, ours);
    row("kalman-cpp-sqrt", ours_version, ours_sqrt);
    row("opencv", CV_VERSION, opencv);
    row("mherb-kalman", KALMAN_MHERB_COMMIT, mherb);
}
