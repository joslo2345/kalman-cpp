#pragma once

/// Library version. The single source of truth: CMake reads these macros to
/// set the project and package version.
#define KALMAN_VERSION_MAJOR 0
#define KALMAN_VERSION_MINOR 1
#define KALMAN_VERSION_PATCH 0
#define KALMAN_VERSION_STRING "0.1.0"

namespace kalman {

/// The library version as "major.minor.patch".
inline constexpr const char* version = KALMAN_VERSION_STRING;

}  // namespace kalman
