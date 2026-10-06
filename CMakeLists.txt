cmake_minimum_required(VERSION 3.20)

# The version lives in include/kalman/version.hpp.
file(STRINGS include/kalman/version.hpp _kalman_version_defines REGEX "#define KALMAN_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "KALMAN_VERSION_${_part} ([0-9]+)" _ "${_kalman_version_defines}")
    set(_kalman_version_${_part} ${CMAKE_MATCH_1})
endforeach()
project(kalman_cpp
    VERSION ${_kalman_version_MAJOR}.${_kalman_version_MINOR}.${_kalman_version_PATCH}
    DESCRIPTION "Header-only C++20 Kalman filters"
    HOMEPAGE_URL "https://github.com/joslo2345/kalman-cpp"
    LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(KALMAN_TOP_LEVEL ON)
else()
    set(KALMAN_TOP_LEVEL OFF)
endif()

# Prefer an installed Eigen 3.4 (package managers provide one); download it
# otherwise. Eigen 5's version rule rejects a 3.4 request, so an untested
# major version is never picked up silently.
option(KALMAN_FETCH_EIGEN "Download Eigen 3.4 if it is not installed" ON)
# The user package registry is skipped: Eigen's own CMake registers every
# build tree there, which would tie this build to some other build directory.
find_package(Eigen3 3.4 QUIET NO_MODULE NO_CMAKE_PACKAGE_REGISTRY)
if(Eigen3_FOUND)
    message(STATUS "kalman-cpp: using installed Eigen ${Eigen3_VERSION}")
elseif(KALMAN_FETCH_EIGEN)
    message(STATUS "kalman-cpp: downloading Eigen 3.4.0")
    set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)  # don't let Eigen register this build tree
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Eigen3)
else()
    message(FATAL_ERROR "kalman-cpp needs Eigen 3.4: install it or set KALMAN_FETCH_EIGEN=ON")
endif()

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

option(KALMAN_BUILD_TESTS "Build tests" ${KALMAN_TOP_LEVEL})
option(KALMAN_WARNINGS_AS_ERRORS "Treat compiler warnings in tests and benchmarks as errors" OFF)
option(KALMAN_SANITIZE "Build tests with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(KALMAN_COVERAGE "Build tests with coverage instrumentation" OFF)

# Warnings and instrumentation for our own targets (tests, benchmarks). Eigen
# and other dependencies are included as system headers so only our code is
# checked.
function(kalman_configure_target target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /bigobj /permissive-)
        if(KALMAN_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
        if(KALMAN_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
    target_include_directories(${target} SYSTEM PRIVATE $<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>)
    if(KALMAN_SANITIZE)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
                                                 -fno-sanitize-recover=undefined)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
    if(KALMAN_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
            target_link_options(${target} PRIVATE -fprofile-instr-generate)
        else()
            target_compile_options(${target} PRIVATE --coverage)
            target_link_options(${target} PRIVATE --coverage)
        endif()
    endif()
endfunction()
option(KALMAN_BUILD_EXAMPLES "Build the examples" ${KALMAN_TOP_LEVEL})
option(KALMAN_BUILD_DOCS "Build the API reference with Doxygen (target: docs)" OFF)
option(KALMAN_BUILD_BENCH "Build benchmarks (needs OpenCV and the mherb/kalman submodule)" OFF)
if(KALMAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(KALMAN_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(KALMAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(KALMAN_BUILD_DOCS)
    add_subdirectory(docs)
endif()
