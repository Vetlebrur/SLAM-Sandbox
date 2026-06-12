#include "stereo_tracker.hpp"
#include "calibration.hpp"
#include <opencv2/highgui.hpp>
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

    StereoTracker tracker(calib);
    DataLoader    loader(bag, 1.0);
    tracker.attach(loader);

    std::cout << "Streaming — press Q in the window to stop\n";
    loader.play();

    return 0;
}
