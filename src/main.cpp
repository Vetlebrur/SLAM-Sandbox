#include "data_loader.hpp"
#include "calibration.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

// ── IMU decode helpers ────────────────────────────────────────────────────────
static inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline double   rd_f64(const uint8_t* p) { double   v; memcpy(&v, p, 8); return v; }

struct ImuData {
    double stamp = 0;
    double ax = 0, ay = 0, az = 0;
    double wx = 0, wy = 0, wz = 0;
};

static ImuData decodeImu(const BagMessage& msg) {
    const uint8_t* d = msg.data.data();
    ImuData imu;
    imu.stamp = rd_u32(d + 4) + rd_u32(d + 8) * 1e-9;
    uint32_t frame_len = rd_u32(d + 12);
    const uint8_t* q = d + 16 + frame_len;   // start of quaternion
    const uint8_t* w = q + 4*8 + 9*8;        // angular velocity
    const uint8_t* a = w + 3*8 + 9*8;        // linear acceleration
    imu.wx = rd_f64(w);     imu.wy = rd_f64(w+8);  imu.wz = rd_f64(w+16);
    imu.ax = rd_f64(a);     imu.ay = rd_f64(a+8);  imu.az = rd_f64(a+16);
    return imu;
}

// ── Overlay text on image ─────────────────────────────────────────────────────
static void putLine(cv::Mat& img, const std::string& text, int row) {
    const cv::Point pos(10, 20 + row * 22);
    const double    scale  = 0.55;
    const int       thick  = 1;
    // dark shadow for readability on any background
    cv::putText(img, text, pos + cv::Point(1,1),
                cv::FONT_HERSHEY_SIMPLEX, scale, {0,0,0}, thick+1);
    cv::putText(img, text, pos,
                cv::FONT_HERSHEY_SIMPLEX, scale, {0,255,0}, thick);
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    const std::string bag =
        "/home/vetlebrur/Projects/github/SLAM-Sandbox/vicon_room1/V1_01_easy/V1_01_easy.bag";
    const std::string mav0 =
        "/home/vetlebrur/Projects/github/SLAM-Sandbox/vicon_room1/V1_01_easy/V1_01_easy/mav0";

    StereoCalib calib = loadEurocCalib(mav0);
    std::cout << "Calibration loaded.\n"
              << "  cam0 K:\n" << calib.cam0.K << "\n"
              << "  stereo baseline: "
              << calib.T_cam1_cam0.translation().norm() << " m\n\n";

    cv::namedWindow("cam0", cv::WINDOW_NORMAL);
    cv::resizeWindow("cam0", 1504, 960);

    ImuData latest_imu;
    int    frame_idx  = 0;
    double t0         = -1.0;  // set on first message

    DataLoader loader(bag, 1.0);

    loader.addCallback("/imu0", [&](const BagMessage& msg) {
        latest_imu = decodeImu(msg);
        if (t0 < 0.0) t0 = latest_imu.stamp;
        latest_imu.stamp -= t0;
    });

    loader.addCallback("/cam0/image_raw", [&](const BagMessage& msg) {
        cv::Mat raw = decodeImage(msg);
        if (raw.empty()) return;

        // Undistort using precomputed maps
        cv::Mat img;
        cv::remap(raw, img, calib.cam0.map1, calib.cam0.map2, cv::INTER_LINEAR);

        // Convert mono to BGR so coloured text shows correctly
        cv::Mat display;
        cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);

        // ── draw your detections on display here ─────────────────────────
        // e.g. cv::circle(display, {x, y}, 3, {0,255,0}, -1);

        // ── overlay ──────────────────────────────────────────────────────
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);

        ss.str(""); ss << "frame: " << frame_idx++;
        putLine(display, ss.str(), 0);

        ss.str(""); ss << "t:  " << latest_imu.stamp;
        putLine(display, ss.str(), 1);

        ss.str(""); ss << "acc: (" << latest_imu.ax << ", "
                                   << latest_imu.ay << ", "
                                   << latest_imu.az << ") m/s²";
        putLine(display, ss.str(), 2);

        ss.str(""); ss << "gyr: (" << latest_imu.wx << ", "
                                   << latest_imu.wy << ", "
                                   << latest_imu.wz << ") rad/s";
        putLine(display, ss.str(), 3);

        cv::imshow("cam0", display);
        cv::waitKey(1);
    });

    std::cout << "Streaming — press Q in the window to stop\n";
    loader.play();

    return 0;
}
