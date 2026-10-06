import os
import re

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.files import copy, load
from conan.tools.layout import basic_layout

required_conan_version = ">=2.0"


class KalmanCppConan(ConanFile):
    name = "kalman-cpp"
    description = "Header-only C++20 Kalman filters: compile-time checked, automatic Jacobians, numerically robust"
    license = "MIT"
    url = "https://github.com/joslo2345/kalman-cpp"
    homepage = "https://github.com/joslo2345/kalman-cpp"
    topics = ("kalman-filter", "state-estimation", "sensor-fusion", "eigen", "header-only")
    package_type = "header-library"
    settings = "os", "arch", "compiler", "build_type"
    exports_sources = "include/*", "LICENSE"
    no_copy_source = True

    def set_version(self):
        # include/kalman/version.hpp is the single source of the version.
        header = load(self, os.path.join(self.recipe_folder, "include", "kalman", "version.hpp"))
        self.version = re.search(r'#define KALMAN_VERSION_STRING "([^"]+)"', header).group(1)

    def layout(self):
        basic_layout(self, src_folder=".")

    def requirements(self):
        self.requires("eigen/3.4.0")

    def package_id(self):
        self.info.clear()

    def validate(self):
        check_min_cppstd(self, 20)

    def package(self):
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))
        copy(self, "*.hpp", os.path.join(self.source_folder, "include"), os.path.join(self.package_folder, "include"))

    def package_info(self):
        # Same names as the library's own CMake package: find_package(kalman), kalman::kalman.
        self.cpp_info.set_property("cmake_file_name", "kalman")
        self.cpp_info.set_property("cmake_target_name", "kalman::kalman")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.requires = ["eigen::eigen"]
