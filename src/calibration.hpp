#pragma once

#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <string>

struct CameraCalib {
    cv::Mat      K;      // 3×3 intrinsic matrix
    cv::Mat      dist;   // distortion coefficients (1×4, radial-tangential)
    Sophus::SE3d T_BS;   // camera-to-body transform
    cv::Mat      map1;   // precomputed undistort remap (x)
    cv::Mat      map2;   // precomputed undistort remap (y)
};

struct ImuCalib {
    Sophus::SE3d T_BS;
    double gyro_noise_density;   // [rad/s/sqrt(Hz)]
    double gyro_random_walk;     // [rad/s^2/sqrt(Hz)]
    double accel_noise_density;  // [m/s^2/sqrt(Hz)]
    double accel_random_walk;    // [m/s^3/sqrt(Hz)]
};

struct StereoCalib {
    CameraCalib  cam0;
    CameraCalib  cam1;
    ImuCalib     imu;
    Sophus::SE3d T_cam1_cam0;  // transforms a point from cam0 frame into cam1 frame
};

// Load calibration from an EuRoC mav0/ directory.
// Expects: <mav0_path>/cam0/sensor.yaml
//          <mav0_path>/cam1/sensor.yaml
//          <mav0_path>/imu0/sensor.yaml
StereoCalib loadEurocCalib(const std::string& mav0_path);
