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
    unit/test_ukf.cpp
    unit/test_smoother.cpp
    unit/test_fusion.cpp
    unit/test_error_state.cpp
    unit/test_diagnostics.cpp
    unit/test_vectors.cpp
    unit/test_small_cholesky.cpp
    unit/test_sqrt_linear.cpp)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(unit_tests PRIVATE KALMAN_VECTORS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vectors")
target_link_libraries(unit_tests PRIVATE kalman::kalman Catch2::Catch2WithMain)
catch_discover_tests(unit_tests)

# OpenCV is an optional, test-only dependency for comparison tests.
find_package(OpenCV QUIET COMPONENTS core video)
if(OpenCV_FOUND)
    add_executable(comparison_tests
        comparison/test_equivalence.cpp
        comparison/test_stability.cpp
        comparison/test_nonlinear.cpp)
    target_include_directories(comparison_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(comparison_tests PRIVATE kalman::kalman Catch2::Catch2WithMain ${OpenCV_LIBS})
    catch_discover_tests(comparison_tests)
else()
    message(STATUS "OpenCV not found: comparison tests disabled")
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
