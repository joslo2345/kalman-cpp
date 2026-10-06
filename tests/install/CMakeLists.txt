# Consumer of an installed kalman-cpp: checks that find_package(kalman) works.
#   cmake --install <build> --prefix <prefix>
#   cmake -S tests/install -B <dir> -DCMAKE_PREFIX_PATH=<prefix> && cmake --build <dir> && <dir>/consumer
cmake_minimum_required(VERSION 3.20)
project(kalman_consumer LANGUAGES CXX)

find_package(kalman 0.1 REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE kalman::kalman)
