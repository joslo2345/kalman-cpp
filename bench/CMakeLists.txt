# Benchmarks against OpenCV and mherb/kalman (guide Step 8). Build in Release:
#   cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DKALMAN_BUILD_BENCH=ON

find_package(OpenCV REQUIRED COMPONENTS core video tracking)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(benchmark)

set(MHERB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/baselines/kalman)
if(NOT EXISTS ${MHERB_DIR}/include/kalman/ExtendedKalmanFilter.hpp)
    message(FATAL_ERROR "mherb/kalman submodule missing: run git submodule update --init")
endif()
execute_process(COMMAND git -C ${MHERB_DIR} rev-parse --short HEAD
    OUTPUT_VARIABLE MHERB_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)

# Same flags for every library; they are recorded next to the results table.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(KALMAN_BENCH_FLAGS -O2 -mcpu=native)
else()
    set(KALMAN_BENCH_FLAGS -O2 -march=native)
endif()

add_library(bench_common INTERFACE)
target_include_directories(bench_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests ${OpenCV_INCLUDE_DIRS})
target_include_directories(bench_common SYSTEM INTERFACE ${MHERB_DIR}/include)
target_compile_definitions(bench_common INTERFACE
    KALMAN_VECTORS_DIR="${PROJECT_SOURCE_DIR}/tests/vectors"
    KALMAN_MHERB_COMMIT="${MHERB_COMMIT}")
target_compile_options(bench_common INTERFACE ${KALMAN_BENCH_FLAGS})
target_link_libraries(bench_common INTERFACE kalman::kalman ${OpenCV_LIBS})

add_executable(bench bench_main.cpp)
target_link_libraries(bench PRIVATE bench_common benchmark::benchmark)

add_executable(accuracy accuracy_main.cpp)
target_link_libraries(accuracy PRIVATE bench_common)

if(APPLE)
    add_library(alloc_counter SHARED alloc_counter.cpp)
    add_executable(bench_alloc bench_alloc.cpp)
    target_compile_options(bench_alloc PRIVATE ${KALMAN_BENCH_FLAGS})
    target_link_libraries(bench_alloc PRIVATE bench_common alloc_counter)
else()
    message(STATUS "bench_alloc uses dyld interposition and is macOS-only; use heaptrack on Linux")
endif()
