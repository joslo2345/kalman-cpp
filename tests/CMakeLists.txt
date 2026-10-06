FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)
include(Catch)

add_executable(unit_tests
    unit/test_smoke.cpp
    unit/test_linear.cpp
    unit/test_autodiff.cpp
    unit/test_ekf.cpp
    unit/test_ukf.cpp)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unit_tests PRIVATE kalman::kalman Catch2::Catch2WithMain)
catch_discover_tests(unit_tests)

# OpenCV is an optional, test-only dependency for comparison tests.
find_package(OpenCV QUIET COMPONENTS core video)
if(OpenCV_FOUND)
    # Add comparison/*.cpp here once they exist (see guide Step 7).
endif()

# Code that must NOT compile. Each target is excluded from the normal build and
# registered as a test that passes only when building it fails.
foreach(name dimension_mismatch measurement_size_mismatch no_jacobian)
    add_executable(compile_fail_${name} EXCLUDE_FROM_ALL compile_fail/${name}.cpp)
    target_link_libraries(compile_fail_${name} PRIVATE kalman::kalman)
    add_test(NAME compile_fail_${name}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target compile_fail_${name})
    set_tests_properties(compile_fail_${name} PROPERTIES WILL_FAIL TRUE)
endforeach()
