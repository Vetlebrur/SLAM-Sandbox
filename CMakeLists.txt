cmake_minimum_required(VERSION 3.16)
project(SLAMSandbox CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui features2d)

# Sophus — header-only Lie groups (SE3, SO3, etc.)
include(FetchContent)
FetchContent_Declare(sophus
    GIT_REPOSITORY https://github.com/strasdat/Sophus.git
    GIT_TAG        1.22.10
    GIT_SHALLOW    TRUE
)
set(BUILD_SOPHUS_TESTS    OFF CACHE BOOL "" FORCE)
set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sophus)

add_subdirectory(src)
