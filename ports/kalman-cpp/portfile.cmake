vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO joslo2345/kalman-cpp
    REF "v${VERSION}"
    SHA512 8aea09158bd4aa86a7b363075e7db09b406cbc15b20a09d349aee9b29bfd3f38fd8ddbefe1da513f092e7f8aeba1a8a25b8e245f6b895122a753b899ef70f991
    HEAD_REF main
)

# Header-only: one (release) configuration is enough.
set(VCPKG_BUILD_TYPE release)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DKALMAN_BUILD_TESTS=OFF
        -DKALMAN_BUILD_EXAMPLES=OFF
        -DKALMAN_FETCH_EIGEN=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME kalman CONFIG_PATH lib/cmake/kalman)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
