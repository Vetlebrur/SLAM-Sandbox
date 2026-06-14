#pragma once

#include "stereo_tracker.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Dense>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string>

// ── Depth → BGR gradient (green=close, red=far) ────────────────────────────────
static cv::Scalar depthColor(float Z, float z_near = 0.5f, float z_far = 7.f) {
    float t = std::clamp((Z - z_near) / (z_far - z_near), 0.f, 1.f);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((int)((1.f - t) * 60), 255, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

static void putLine(cv::Mat& img, const std::string& text, int row) {
    const cv::Point pos(10, 20 + row * 22);
    cv::putText(img, text, pos + cv::Point(1,1),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 2);
    cv::putText(img, text, pos,
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,0}, 1);
}

// ── Visualizer ────────────────────────────────────────────────────────────────
// Subscribed to StereoTracker's frame callback and updated by main after each
// EKF update. Owns the OpenCV windows and TUM trajectory file.
class Visualizer {
public:
    explicit Visualizer(std::string tum_path = "") : tum_path_(std::move(tum_path)) {}

    // Called from main's pose callback, after EKF has been updated
    void updateBackend(double stamp,
                       const Sophus::SE3d& ekf_pose,
                       const Eigen::Vector3d& ekf_vel,
                       const Sophus::SE3d& pnp_pose,
                       size_t n_tracked)
    {
        ekf_pose_  = ekf_pose;
        ekf_vel_   = ekf_vel;
        pnp_pose_  = pnp_pose;
        n_tracked_ = n_tracked;
        if (!tum_path_.empty()) writeTUM_(stamp, ekf_pose);
    }

    // Called from main's frame callback (StereoTracker fires this every cam0)
    void render(const FrameData& f) {
        if (f.cam0.empty()) return;

        cv::Mat display;
        cv::cvtColor(f.cam0, display, cv::COLOR_GRAY2BGR);

        for (int i = 0; i < (int)f.kp.size(); i++) {
            auto it = f.depth.find(i);
            cv::Scalar colour(80, 80, 80);
            if (it != f.depth.end()) colour = depthColor(it->second);
            cv::circle(display, f.kp[i].pt, 4, colour, -1);
        }

        for (auto& a : f.flow) {
            cv::Scalar colour = a.inlier ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 0, 200);
            cv::arrowedLine(display, a.p0, a.p1, colour, 1, cv::LINE_AA, 0, 0.4);
            if (a.inlier) cv::circle(display, a.p1, 2, {255, 255, 0}, -1);
        }

        const Eigen::Vector3d t_pnp = pnp_pose_.translation();
        const Eigen::Vector3d t_ekf = ekf_pose_.translation();
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss.str(""); ss << "frame: "   << f.frame_idx;
        putLine(display, ss.str(), 0);
        ss.str(""); ss << "t: "       << f.stamp;
        putLine(display, ss.str(), 1);
        ss.str(""); ss << "tracked: " << n_tracked_;
        putLine(display, ss.str(), 2);
        ss.str(""); ss << "pnp: (" << t_pnp.x() << ", " << t_pnp.y() << ", " << t_pnp.z() << ")";
        putLine(display, ss.str(), 3);
        ss.str(""); ss << "ekf: (" << t_ekf.x() << ", " << t_ekf.y() << ", " << t_ekf.z() << ")";
        putLine(display, ss.str(), 4);
        ss.str(""); ss << "vel: (" << ekf_vel_.x() << ", " << ekf_vel_.y() << ", " << ekf_vel_.z() << ") m/s";
        putLine(display, ss.str(), 5);

        cv::imshow("cam0", display);
        if (!f.stereo.empty()) cv::imshow("stereo", f.stereo);
        cv::waitKey(1);
    }

private:
    void writeTUM_(double stamp, const Sophus::SE3d& T) {
        std::ofstream f(tum_path_, std::ios::app);
        auto t = T.translation();
        auto q = T.unit_quaternion();
        f << std::fixed << std::setprecision(9)
          << stamp << " "
          << t.x() << " " << t.y() << " " << t.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }

    std::string     tum_path_;
    Sophus::SE3d    ekf_pose_;
    Eigen::Vector3d ekf_vel_{0, 0, 0};
    Sophus::SE3d    pnp_pose_;
    size_t          n_tracked_ = 0;
};
