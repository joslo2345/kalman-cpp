#pragma once

// Loader for the frozen benchmark scenarios in tests/vectors/ (see
// scripts/make_vectors.py for the format). KALMAN_VECTORS_DIR is set by CMake.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace scenarios {

struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<double> data;  // float32 files are widened exactly

    std::size_t size() const {
        std::size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

// Minimal reader for little-endian, C-order float32/float64 .npy files.
inline NpyArray load_npy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    char magic[6];
    in.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not a .npy file: " + path);
    unsigned char version[2];
    in.read(reinterpret_cast<char*>(version), 2);
    std::uint32_t header_len = 0;
    if (version[0] == 1) {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        header_len = b[0] | (b[1] << 8);
    } else {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        header_len = b[0] | (b[1] << 8) | (b[2] << 16) | (std::uint32_t(b[3]) << 24);
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);

    auto field = [&](const std::string& key) {
        const auto pos = header.find("'" + key + "'");
        if (pos == std::string::npos) throw std::runtime_error("bad .npy header in " + path);
        return header.substr(header.find(':', pos) + 1);
    };
    const std::string descr = field("descr");
    const bool f8 = descr.find("<f8") != std::string::npos;
    const bool f4 = descr.find("<f4") != std::string::npos;
    if (!f8 && !f4) throw std::runtime_error("unsupported dtype in " + path);
    if (field("fortran_order").find("False") == std::string::npos) {
        throw std::runtime_error("Fortran order not supported: " + path);
    }

    NpyArray out;
    const std::string shape = field("shape");
    const std::string dims = shape.substr(shape.find('(') + 1, shape.find(')') - shape.find('(') - 1);
    std::size_t start = 0;
    while (start < dims.size()) {
        const std::size_t end = dims.find(',', start);
        const std::string tok = dims.substr(start, end - start);
        if (tok.find_first_of("0123456789") != std::string::npos) out.shape.push_back(std::stoull(tok));
        if (end == std::string::npos) break;
        start = end + 1;
    }

    out.data.resize(out.size());
    if (f8) {
        in.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(out.size() * 8));
    } else {
        std::vector<float> tmp(out.size());
        in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(out.size() * 4));
        for (std::size_t i = 0; i < tmp.size(); ++i) out.data[i] = tmp[i];
    }
    if (!in) throw std::runtime_error("truncated .npy file: " + path);
    return out;
}

inline std::string vectors_path(const std::string& scenario, const std::string& array) {
    return std::string(KALMAN_VECTORS_DIR) + "/" + scenario + "/" + array + ".npy";
}

// A matrix (R, C), a vector (R,), or a scalar () as a 1x1 matrix.
template <int R, int C, typename Scalar = double>
Eigen::Matrix<Scalar, R, C> as_matrix(const NpyArray& a, const std::string& what) {
    const bool ok = (a.shape.size() == 2 && a.shape[0] == std::size_t(R) && a.shape[1] == std::size_t(C)) ||
                    (C == 1 && a.shape.size() == 1 && a.shape[0] == std::size_t(R)) ||
                    (R == 1 && C == 1 && a.shape.empty());
    if (!ok) throw std::runtime_error("unexpected shape for " + what);
    Eigen::Matrix<Scalar, R, C> m;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) m(r, c) = static_cast<Scalar>(a.data[r * C + c]);
    return m;
}

// Rows of the last two dimensions, starting at flat row `first_row`.
template <int N, typename Scalar = double>
std::vector<Eigen::Matrix<Scalar, N, 1>> as_rows(const NpyArray& a, std::size_t first_row, std::size_t count) {
    if (a.shape.empty() || a.shape.back() != std::size_t(N)) throw std::runtime_error("unexpected row size");
    std::vector<Eigen::Matrix<Scalar, N, 1>> out(count);
    for (std::size_t k = 0; k < count; ++k)
        for (int i = 0; i < N; ++i) out[k](i) = static_cast<Scalar>(a.data[(first_row + k) * N + i]);
    return out;
}

template <int N, int M, typename Scalar = double>
struct Scenario {
    Eigen::Matrix<Scalar, N, N> F, Q, P0;
    Eigen::Matrix<Scalar, M, N> H;
    Eigen::Matrix<Scalar, M, M> R;
    Eigen::Matrix<Scalar, N, 1> x0;
    Scalar dt;
    std::vector<Eigen::Matrix<Scalar, N, 1>> truth;  // empty for S4
    std::vector<Eigen::Matrix<Scalar, M, 1>> zs;
};

// Linear scenarios: S1 (2/1), S2 (4/2), S4 (4/2, float), S5 (15/6).
template <int N, int M, typename Scalar = double>
Scenario<N, M, Scalar> load(const std::string& name) {
    Scenario<N, M, Scalar> sc;
    sc.F = as_matrix<N, N, Scalar>(load_npy(vectors_path(name, "F")), name + "/F");
    sc.H = as_matrix<M, N, Scalar>(load_npy(vectors_path(name, "H")), name + "/H");
    sc.Q = as_matrix<N, N, Scalar>(load_npy(vectors_path(name, "Q")), name + "/Q");
    sc.R = as_matrix<M, M, Scalar>(load_npy(vectors_path(name, "R")), name + "/R");
    sc.x0 = as_matrix<N, 1, Scalar>(load_npy(vectors_path(name, "x0")), name + "/x0");
    sc.P0 = as_matrix<N, N, Scalar>(load_npy(vectors_path(name, "P0")), name + "/P0");
    sc.dt = as_matrix<1, 1, Scalar>(load_npy(vectors_path(name, "dt")), name + "/dt")(0, 0);
    const NpyArray z = load_npy(vectors_path(name, "z"));
    sc.zs = as_rows<M, Scalar>(z, 0, z.shape[0]);
    if (name != "S4") {
        const NpyArray t = load_npy(vectors_path(name, "truth"));
        sc.truth = as_rows<N, Scalar>(t, 0, t.shape[0]);
    }
    return sc;
}

// S3: range-bearing tracking, many seeds sharing one model.
struct MultiSeedScenario {
    Eigen::Matrix4d F, Q, P0;
    Eigen::Matrix2d R;
    Eigen::Vector4d x0;
    double dt;
    std::vector<std::vector<Eigen::Vector4d>> truth;  // [seed][step]
    std::vector<std::vector<Eigen::Vector2d>> zs;     // [seed][step]
};

inline MultiSeedScenario load_range_bearing(const std::string& name = "S3") {
    MultiSeedScenario sc;
    sc.F = as_matrix<4, 4>(load_npy(vectors_path(name, "F")), name + "/F");
    sc.Q = as_matrix<4, 4>(load_npy(vectors_path(name, "Q")), name + "/Q");
    sc.R = as_matrix<2, 2>(load_npy(vectors_path(name, "R")), name + "/R");
    sc.x0 = as_matrix<4, 1>(load_npy(vectors_path(name, "x0")), name + "/x0");
    sc.P0 = as_matrix<4, 4>(load_npy(vectors_path(name, "P0")), name + "/P0");
    sc.dt = as_matrix<1, 1>(load_npy(vectors_path(name, "dt")), name + "/dt")(0, 0);
    const NpyArray t = load_npy(vectors_path(name, "truth"));
    const NpyArray z = load_npy(vectors_path(name, "z"));
    if (t.shape.size() != 3 || z.shape.size() != 3) throw std::runtime_error("S3 arrays must be 3-D");
    const std::size_t seeds = t.shape[0], steps = t.shape[1];
    for (std::size_t s = 0; s < seeds; ++s) {
        sc.truth.push_back(as_rows<4>(t, s * steps, steps));
        sc.zs.push_back(as_rows<2>(z, s * steps, steps));
    }
    return sc;
}

}  // namespace scenarios
