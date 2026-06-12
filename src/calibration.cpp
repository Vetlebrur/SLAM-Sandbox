#include "calibration.hpp"

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>
#include <stdexcept>

static Sophus::SE3d parseTBS(const YAML::Node& node) {
    auto data = node["T_BS"]["data"].as<std::vector<double>>();
    if (data.size() != 16)
        throw std::runtime_error("T_BS data should have 16 elements");

    Eigen::Matrix4d M;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            M(r, c) = data[r * 4 + c];

    return Sophus::SE3d(M);
}

static CameraCalib loadCamera(const std::string& yaml_path) {
    YAML::Node n = YAML::LoadFile(yaml_path);

    // Intrinsics [fu, fv, cu, cv]
    auto intr = n["intrinsics"].as<std::vector<double>>();
    cv::Mat K = (cv::Mat_<double>(3, 3)
        << intr[0], 0,       intr[2],
           0,       intr[1], intr[3],
           0,       0,       1);

    // Distortion coefficients
    auto dv = n["distortion_coefficients"].as<std::vector<double>>();
    cv::Mat dist(1, static_cast<int>(dv.size()), CV_64F, dv.data());
    dist = dist.clone();

    // Image size
    auto res = n["resolution"].as<std::vector<int>>();
    cv::Size sz(res[0], res[1]);

    Sophus::SE3d T_BS = parseTBS(n);

    // Precompute undistort/rectify maps (output uses same K, no zoom)
    CameraCalib cam;
    cam.K    = K;
    cam.dist = dist;
    cam.T_BS = T_BS;
    cv::initUndistortRectifyMap(K, dist, cv::noArray(), K, sz,
                                CV_32FC1, cam.map1, cam.map2);
    return cam;
}

static ImuCalib loadImu(const std::string& yaml_path) {
    YAML::Node n = YAML::LoadFile(yaml_path);

    ImuCalib imu;
    imu.T_BS               = parseTBS(n);
    imu.gyro_noise_density  = n["gyroscope_noise_density"].as<double>();
    imu.gyro_random_walk    = n["gyroscope_random_walk"].as<double>();
    imu.accel_noise_density = n["accelerometer_noise_density"].as<double>();
    imu.accel_random_walk   = n["accelerometer_random_walk"].as<double>();
    return imu;
}

StereoCalib loadEurocCalib(const std::string& mav0_path) {
    StereoCalib s;
    s.cam0 = loadCamera(mav0_path + "/cam0/sensor.yaml");
    s.cam1 = loadCamera(mav0_path + "/cam1/sensor.yaml");
    s.imu  = loadImu   (mav0_path + "/imu0/sensor.yaml");

    // T_cam1_cam0 = T_BS_cam1^-1 * T_BS_cam0
    s.T_cam1_cam0 = s.cam1.T_BS.inverse() * s.cam0.T_BS;
    return s;
}
