find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)

# bag_reader — utility library built from cpp/
add_library(bag_reader STATIC ${CMAKE_SOURCE_DIR}/cpp/bag_reader.cpp)
target_include_directories(bag_reader PUBLIC  ${CMAKE_SOURCE_DIR}/cpp ${OpenCV_INCLUDE_DIRS})
target_link_libraries     (bag_reader PRIVATE ${LZ4_LIBRARIES} opencv_core opencv_imgproc opencv_highgui opencv_features2d)
target_include_directories(bag_reader PRIVATE ${LZ4_INCLUDE_DIRS})

# calibration — utility library built from src/
find_package(yaml-cpp REQUIRED)

add_library(calibration STATIC calibration.cpp)
target_include_directories(calibration PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries     (calibration PRIVATE opencv_core opencv_calib3d Sophus::Sophus Eigen3::Eigen yaml-cpp::yaml-cpp)

# Build every other .cpp as a standalone executable
file(GLOB SOURCES "*.cpp")
foreach(SRC ${SOURCES})
    get_filename_component(NAME ${SRC} NAME_WE)
    if(NAME STREQUAL "bag_reader" OR NAME STREQUAL "calibration")
        continue()
    endif()
    add_executable(${NAME} ${SRC})
    target_link_libraries(${NAME} PRIVATE
        bag_reader
        calibration
        Sophus::Sophus
        Ceres::ceres
        Eigen3::Eigen
        opencv_core opencv_imgproc opencv_highgui opencv_features2d opencv_calib3d
    )
    target_include_directories(${NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()
