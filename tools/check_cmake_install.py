"""Build a downstream consumer against the installed CMake package."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(deform3d_consumer LANGUAGES CXX)
find_package(deform3d CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE deform3d::deform3d)
enable_testing()
add_test(NAME downstream COMMAND consumer)
''')
        (root / "main.cpp").write_text('''#include <deform3d/deform3d.hpp>
#include <deform3d/embed.hpp>
#include <deform3d/simplex_material.hpp>
int main() { return deform3d::SimplexMaterial<double>::endpoint(3, 0, 1) == 1 ? 0 : 1; }
''')
        subprocess.run(["cmake", "-S", directory, "-B", str(root / "build"),
                        f"-DCMAKE_PREFIX_PATH={args.prefix.resolve()}"], check=True)
        subprocess.run(["cmake", "--build", str(root / "build"), "--config", "Release",
                        "--parallel", "2"], check=True)
        subprocess.run(["ctest", "--test-dir", str(root / "build"), "-C", "Release",
                        "--output-on-failure"], check=True)


if __name__ == "__main__":
    main()
