#include <string>

#include <catch2/catch_test_macros.hpp>
#include <kalman/kalman.hpp>

TEST_CASE("Version macros, string and constant agree", "[unit][version]") {
    const std::string expected = std::to_string(KALMAN_VERSION_MAJOR) + "." + std::to_string(KALMAN_VERSION_MINOR) +
                                 "." + std::to_string(KALMAN_VERSION_PATCH);
    REQUIRE(expected == KALMAN_VERSION_STRING);
    REQUIRE(expected == kalman::version);
}
