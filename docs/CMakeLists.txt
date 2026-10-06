# API reference (guide Step 10). Needs Doxygen; the doxygen-awesome theme is fetched.
find_package(Doxygen 1.9 REQUIRED)

FetchContent_Declare(doxygen_awesome
    GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG v2.5.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doxygen_awesome)

option(KALMAN_DOCS_WARN_AS_ERROR "Fail the docs build on Doxygen warnings" OFF)
if(KALMAN_DOCS_WARN_AS_ERROR)
    set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
else()
    set(DOXYGEN_WARN_AS_ERROR NO)
endif()
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(AWESOME_CSS ${doxygen_awesome_SOURCE_DIR}/doxygen-awesome.css)
set(AWESOME_SIDEBAR_CSS ${doxygen_awesome_SOURCE_DIR}/doxygen-awesome-sidebar-only.css)
configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Building the API reference in ${CMAKE_CURRENT_BINARY_DIR}/html"
    VERBATIM)
