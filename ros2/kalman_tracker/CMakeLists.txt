cmake_minimum_required(VERSION 3.20)
project(kalman_tracker LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(kalman 0.1 REQUIRED)

add_executable(tracker_node src/tracker_node.cpp)
target_compile_features(tracker_node PRIVATE cxx_std_20)
target_link_libraries(tracker_node PRIVATE
    kalman::kalman rclcpp::rclcpp ${geometry_msgs_TARGETS} ${nav_msgs_TARGETS})

install(TARGETS tracker_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
