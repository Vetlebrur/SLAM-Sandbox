find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)

# ── test_eigen ──────────────────────────────────────────────────────────────
add_executable(test_eigen test_eigen.cpp)
target_link_libraries(test_eigen PRIVATE Eigen3::Eigen)

# ── test_ceres ──────────────────────────────────────────────────────────────
add_executable(test_ceres test_ceres.cpp)
target_link_libraries(test_ceres PRIVATE Ceres::ceres)

# ── test_sophus ─────────────────────────────────────────────────────────────
add_executable(test_sophus test_sophus.cpp)
target_link_libraries(test_sophus PRIVATE Sophus::Sophus)

# ── bag_reader (static lib) ──────────────────────────────────────────────────
add_library(bag_reader STATIC bag_reader.cpp)
target_include_directories(bag_reader PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries     (bag_reader PRIVATE ${LZ4_LIBRARIES})
target_include_directories(bag_reader PRIVATE ${LZ4_INCLUDE_DIRS})

# ── hello_world ──────────────────────────────────────────────────────────────
add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world PRIVATE Sophus::Sophus Ceres::ceres)

# ── test_bag ─────────────────────────────────────────────────────────────────
add_executable(test_bag test_bag.cpp)
target_link_libraries(test_bag PRIVATE bag_reader Eigen3::Eigen)
