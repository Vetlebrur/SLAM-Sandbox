cmake_minimum_required(VERSION 3.16)
project(SLAMSandbox CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Flatpak (VS Code extension) mounts the host filesystem at /run/host.
# Host binaries in /run/host/usr/bin can't execute inside the sandbox, so pin
# GIT_EXECUTABLE to the Flatpak runtime git before FetchContent can discover
# the non-functional host git binary via CMAKE_PREFIX_PATH.
if(EXISTS "/run/host/usr")
    find_program(_flatpak_git git PATHS /usr/bin NO_DEFAULT_PATH)
    if(_flatpak_git)
        set(GIT_EXECUTABLE "${_flatpak_git}" CACHE FILEPATH "Git executable" FORCE)
    endif()
    list(APPEND CMAKE_PREFIX_PATH "/run/host/usr")
    # FindBLAS/FindLAPACK ignore CMAKE_LIBRARY_PATH inside find_dependency() calls,
    # so pre-seed the libraries directly to short-circuit the search.
    set(BLAS_LIBRARIES   "/run/host/usr/lib/libblas.so"   CACHE FILEPATH "" FORCE)
    set(LAPACK_LIBRARIES "/run/host/usr/lib/liblapack.so" CACHE FILEPATH "" FORCE)
endif()
list(APPEND CMAKE_PREFIX_PATH "/usr/share/eigen3")
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui features2d video)

# Sophus — header-only Lie groups (SE3, SO3, etc.)
include(FetchContent)
FetchContent_Declare(sophus
    GIT_REPOSITORY https://github.com/strasdat/Sophus.git
    GIT_TAG        1.22.10
    GIT_SHALLOW    TRUE
)
set(BUILD_SOPHUS_TESTS    OFF CACHE BOOL "" FORCE)
set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)
FetchContent_MakeAvailable(sophus)

add_subdirectory(src)
