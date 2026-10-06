cmake_minimum_required(VERSION 3.20)
project(test_package LANGUAGES CXX)

find_package(kalman REQUIRED CONFIG)

add_executable(consumer test_package.cpp)
target_link_libraries(consumer PRIVATE kalman::kalman)
target_compile_features(consumer PRIVATE cxx_std_20)
