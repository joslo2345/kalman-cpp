cmake_minimum_required(VERSION 3.20)
project(kalman_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Eigen3)

add_library(kalman INTERFACE)
add_library(kalman::kalman ALIAS kalman)
target_include_directories(kalman INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(kalman INTERFACE Eigen3::Eigen)
target_compile_features(kalman INTERFACE cxx_std_20)

# Install / export so users can call find_package(kalman)
install(TARGETS kalman EXPORT kalmanTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT kalmanTargets
    NAMESPACE kalman::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kalman)
configure_package_config_file(cmake/kalmanConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kalman)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/kalmanConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/kalmanConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kalman)
export(EXPORT kalmanTargets
    NAMESPACE kalman::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/kalmanTargets.cmake)

option(KALMAN_BUILD_TESTS "Build tests" ON)
option(KALMAN_BUILD_BENCH "Build benchmarks (needs OpenCV and the mherb/kalman submodule)" OFF)
if(KALMAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(KALMAN_BUILD_BENCH)
    add_subdirectory(bench)
endif()
