#pragma once

#include "calibration.hpp"
#include "data_loader.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <map>

// ── IMU ───────────────────────────────────────────────────────────────────────
struct ImuData {
    double stamp = 0;
    double ax = 0, ay = 0, az = 0;
    double wx = 0, wy = 0, wz = 0;
};

static inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline double   rd_f64(const uint8_t* p) { double   v; memcpy(&v, p, 8); return v; }

static ImuData decodeImu(const BagMessage& msg) {
    const uint8_t* d = msg.data.data();
    ImuData imu;
    imu.stamp = rd_u32(d + 4) + rd_u32(d + 8) * 1e-9;
    uint32_t frame_len = rd_u32(d + 12);
    const uint8_t* q = d + 16 + frame_len;
    const uint8_t* w = q + 4*8 + 9*8;
    const uint8_t* a = w + 3*8 + 9*8;
    imu.wx = rd_f64(w);   imu.wy = rd_f64(w+8);  imu.wz = rd_f64(w+16);
    imu.ax = rd_f64(a);   imu.ay = rd_f64(a+8);  imu.az = rd_f64(a+16);
    return imu;
}

// ── Text overlay ──────────────────────────────────────────────────────────────
static void putLine(cv::Mat& img, const std::string& text, int row) {
    const cv::Point pos(10, 20 + row * 22);
    cv::putText(img, text, pos + cv::Point(1,1),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 2);
    cv::putText(img, text, pos,
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,0}, 1);
}

// ── Depth → BGR gradient (green=close, red=far) ───────────────────────────────
static cv::Scalar depthColor(float Z, float z_near = 0.5f, float z_far = 7.f) {
    float t = std::clamp((Z - z_near) / (z_far - z_near), 0.f, 1.f);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((int)((1.f - t) * 60), 255, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

// ── StereoTracker ─────────────────────────────────────────────────────────────
class StereoTracker {
public:
    explicit StereoTracker(const StereoCalib& calib, int max_features = 500)
        : calib_(calib), orb_(cv::ORB::create(max_features))
    {
        P0_ = cv::Mat::zeros(3, 4, CV_64F);
        calib_.cam0.K.copyTo(P0_(cv::Rect(0, 0, 3, 3)));

        Eigen::Matrix4d T = calib_.T_cam1_cam0.matrix();
        cv::Mat Rt(3, 4, CV_64F);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                Rt.at<double>(r, c) = T(r, c);
        P1_ = calib_.cam1.K * Rt;
    }

    void attach(DataLoader& loader) {
        loader.addCallback("/imu0",          [this](const BagMessage& m){ onImu(m); });
        loader.addCallback("/cam0/image_raw", [this](const BagMessage& m){ onCam0(m); });
        loader.addCallback("/cam1/image_raw", [this](const BagMessage& m){ onCam1(m); });
    }

private:
    // ── Per-camera pending frame ───────────────────────────────────────────────
    struct Frame {
        cv::Mat                   img;
        std::vector<cv::KeyPoint> kp;
        cv::Mat                   desc;
        double                    stamp = -1.0;

        bool ready() const { return stamp >= 0.0; }
        void clear() { stamp = -1.0; kp.clear(); desc.release(); }
    };

    const StereoCalib& calib_;
    cv::Ptr<cv::ORB>   orb_;

    Frame   f0_, f1_;       // pending frames — filled by callbacks, consumed by tryProcess
    ImuData latest_imu_;
    double  t0_        = -1.0;
    int     frame_idx_ = 0;

    cv::Mat P0_, P1_;

    // Last successfully triangulated keypoints + depths — updated by processPair,
    // rendered every cam0 frame so the image stays live between stereo matches.
    std::vector<cv::KeyPoint> display_kp_;
    std::map<int, float>      display_depth_;

    // ── IMU callback ──────────────────────────────────────────────────────────
    void onImu(const BagMessage& msg) {
        latest_imu_ = decodeImu(msg);
        if (t0_ < 0.0) t0_ = latest_imu_.stamp;
        latest_imu_.stamp -= t0_;
    }

    // ── cam0: undistort + detect, render live image with last known dots ─────────
    void onCam0(const BagMessage& msg) {
        cv::Mat raw = decodeImage(msg);
        if (raw.empty()) return;
        f0_.stamp = msg.stamp.toSec();
        cv::remap(raw, f0_.img, calib_.cam0.map1, calib_.cam0.map2, cv::INTER_LINEAR);
        orb_->detectAndCompute(f0_.img, cv::noArray(), f0_.kp, f0_.desc);

        cv::Mat display;
        cv::cvtColor(f0_.img, display, cv::COLOR_GRAY2BGR);
        for (int i = 0; i < (int)display_kp_.size(); i++) {
            auto it = display_depth_.find(i);
            cv::Scalar colour(80, 80, 80);
            if (it != display_depth_.end())
                colour = depthColor(it->second);
            cv::circle(display, display_kp_[i].pt, 4, colour, -1);
        }

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss.str(""); ss << "frame: " << frame_idx_++;      putLine(display, ss.str(), 0);
        ss.str(""); ss << "t:  "    << latest_imu_.stamp; putLine(display, ss.str(), 1);
        ss.str(""); ss << "acc: (" << latest_imu_.ax << ", " << latest_imu_.ay << ", " << latest_imu_.az << ") m/s²";
        putLine(display, ss.str(), 2);
        ss.str(""); ss << "gyr: (" << latest_imu_.wx << ", " << latest_imu_.wy << ", " << latest_imu_.wz << ") rad/s";
        putLine(display, ss.str(), 3);

        cv::imshow("cam0", display);
        cv::waitKey(1);

        tryProcess();
    }

    // ── cam1: undistort, then try to pair ─────────────────────────────────────
    void onCam1(const BagMessage& msg) {
        cv::Mat raw = decodeImage(msg);
        if (raw.empty()) return;
        f1_.stamp = msg.stamp.toSec();
        cv::remap(raw, f1_.img, calib_.cam1.map1, calib_.cam1.map2, cv::INTER_LINEAR);
        tryProcess();
    }

    // ── Barrier: runs only when both frames are ready with matching timestamps ─
    void tryProcess() {
        if (!f0_.ready() || !f1_.ready()) return;

        double dt = std::abs(f0_.stamp - f1_.stamp);
        if (dt > 0.04) {
            // Timestamps don't match — discard the older frame and wait for a new one
            if (f0_.stamp < f1_.stamp) f0_.clear();
            else                       f1_.clear();
            return;
        }

        // Matched pair — run the full stereo pipeline
        processPair();

        // Consume both frames so they aren't reused next cycle
        f0_.clear();
        f1_.clear();
    }

    // ── Stereo pipeline: match → triangulate → display ────────────────────────
    void processPair() {
        // Detect ORB in cam1
        std::vector<cv::KeyPoint> kp1;
        cv::Mat desc1;
        orb_->detectAndCompute(f1_.img, cv::noArray(), kp1, desc1);

        // Match cam0 descriptors against cam1
        std::vector<cv::DMatch> matches;
        if (!f0_.desc.empty() && !desc1.empty()) {
            cv::BFMatcher matcher(cv::NORM_HAMMING, /*crossCheck=*/true);
            matcher.match(f0_.desc, desc1, matches);
            std::sort(matches.begin(), matches.end(),
                [](auto& a, auto& b){ return a.distance < b.distance; });
            if (matches.size() > 150) matches.resize(150);
        }

        // Stereo side-by-side view
        cv::Mat stereo;
        cv::drawMatches(f0_.img, f0_.kp, f1_.img, kp1, matches, stereo,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                        cv::DrawMatchesFlags::DEFAULT);
        cv::imshow("stereo", stereo);

        // Triangulate matched points → update display_kp_ / display_depth_
        // (onCam0 will overlay these on the next rendered frame)
        display_depth_.clear();
        if (!matches.empty()) {
            std::vector<cv::Point2f> pts0, pts1;
            for (auto& m : matches) {
                pts0.push_back(f0_.kp[m.queryIdx].pt);
                pts1.push_back(kp1[m.trainIdx].pt);
            }
            cv::Mat pts4D;
            cv::triangulatePoints(P0_, P1_, pts0, pts1, pts4D);
            for (int i = 0; i < (int)matches.size(); i++) {
                float W = pts4D.at<float>(3, i);
                if (std::abs(W) < 1e-9f) continue;
                float Z = pts4D.at<float>(2, i) / W;
                if (Z > 0.f && Z < 10.f)
                    display_depth_[matches[i].queryIdx] = Z;
            }
        }
        display_kp_ = f0_.kp;
    }
};
