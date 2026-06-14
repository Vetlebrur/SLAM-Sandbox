#include "stereo_tracker.hpp"
#include "imu_ekf.hpp"
#include "visualizer.hpp"
#include "calibration.hpp"
#include <iostream>

int main() {
    const std::string bag =
        "/home/vetlebrur/Projects/github/SLAM-Sandbox/vicon_room1/V1_01_easy/V1_01_easy.bag";
    const std::string mav0 =
        "/home/vetlebrur/Projects/github/SLAM-Sandbox/vicon_room1/V1_01_easy/V1_01_easy/mav0";

    StereoCalib calib = loadEurocCalib(mav0);
    std::cout << "Calibration loaded.\n"
              << "  cam0 K:\n" << calib.cam0.K << "\n"
              << "  cam1 K:\n" << calib.cam1.K << "\n"
              << "  stereo baseline: "
              << calib.T_cam1_cam0.translation().norm() << " m\n\n";

    cv::namedWindow("cam0",   cv::WINDOW_NORMAL);
    cv::namedWindow("stereo", cv::WINDOW_NORMAL);
    cv::resizeWindow("cam0",   752,  480);
    cv::resizeWindow("stereo", 1504, 480);

    // ── Backend ───────────────────────────────────────────────────────────────
    ImuEkf    ekf(calib.imu);
    Visualizer viz("trajectory.txt");

    // ── Frontend ──────────────────────────────────────────────────────────────
    StereoTracker tracker(calib);
    DataLoader    loader(bag, 1.0);

    // EKF propagation driven directly from the bag's IMU topic
    loader.addCallback("/imu0", [&](const BagMessage& msg) {
        ImuData imu = decodeImu(msg);
        ekf.propagate(imu.stamp,
                      {imu.ax, imu.ay, imu.az},
                      {imu.wx, imu.wy, imu.wz});
    });

    // When PnP succeeds: update EKF, then hand both poses to the visualizer
    tracker.onPose([&](double stamp, const Sophus::SE3d& T_pnp, size_t n) {
        ekf.update(T_pnp);
        viz.updateBackend(stamp, ekf.pose(), ekf.velocity(), T_pnp, n);
    });

    // Every cam0 frame: render whatever the visualizer has cached
    tracker.onFrame([&](const FrameData& f) {
        viz.render(f);
    });

    tracker.attach(loader);

    std::cout << "Streaming — press Q in the window to stop\n";
    loader.play();

    return 0;
}
