#pragma once

#include "calibration.hpp"
#include "data_loader.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <functional>
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

// ── Data types shared between tracker and visualizer ─────────────────────────
struct FlowArrow { cv::Point2f p0, p1; bool inlier; };

struct FrameData {
    double                    stamp;
    int                       frame_idx;
    cv::Mat                   cam0;       // current gray undistorted
    cv::Mat                   stereo;     // drawMatches from previous pair (may be empty)
    std::vector<cv::KeyPoint> kp;         // cam0 keypoints
    std::map<int, float>      depth;      // kp_idx → depth [m] from previous pair
    std::vector<FlowArrow>    flow;       // KLT flow from previous pair
    size_t                    n_tracked;
    Sophus::SE3d              T_pnp;      // PnP pose from previous pair
};

// ── StereoTracker — visual frontend only ─────────────────────────────────────
class StereoTracker {
public:
    // pose_cb : fires each time PnP succeeds  (stamp, T_world_cam, n_inliers)
    // frame_cb: fires every cam0 frame        (full display snapshot)
    using PoseCallback  = std::function<void(double, const Sophus::SE3d&, size_t)>;
    using FrameCallback = std::function<void(const FrameData&)>;

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

    void onPose (PoseCallback  cb) { pose_cb_  = std::move(cb); }
    void onFrame(FrameCallback cb) { frame_cb_ = std::move(cb); }

    void attach(DataLoader& loader) {
        loader.addCallback("/imu0",           [this](const BagMessage& m){ onImu(m); });
        loader.addCallback("/cam0/image_raw", [this](const BagMessage& m){ onCam0(m); });
        loader.addCallback("/cam1/image_raw", [this](const BagMessage& m){ onCam1(m); });
    }

private:
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

    Frame  f0_, f1_;
    double t0_           = -1.0;
    double latest_stamp_ =  0.0;
    int    frame_idx_    =  0;

    cv::Mat P0_, P1_;

    // Snapshot updated each processPair(), consumed by next frame callback
    std::vector<cv::KeyPoint> display_kp_;
    std::map<int, float>      display_depth_;
    std::vector<FlowArrow>    display_flow_;
    cv::Mat                   display_stereo_;
    Sophus::SE3d              T_world_cam_;

    cv::Mat                  prev_img_;
    std::vector<cv::Point2f> tracked_pts2d_;
    std::vector<cv::Point3f> tracked_pts3d_;

    PoseCallback  pose_cb_;
    FrameCallback frame_cb_;

    static constexpr int kMinTrackPts = 80;

    // ── Normalise timestamps relative to first IMU sample ────────────────────
    void onImu(const BagMessage& msg) {
        ImuData imu = decodeImu(msg);
        if (t0_ < 0.0) t0_ = imu.stamp;
        latest_stamp_ = imu.stamp - t0_;
    }

    // ── cam0: detect + fire frame callback with previous pair's display data ─
    void onCam0(const BagMessage& msg) {
        cv::Mat raw = decodeImage(msg);
        if (raw.empty()) return;
        f0_.stamp = msg.stamp.toSec();
        cv::remap(raw, f0_.img, calib_.cam0.map1, calib_.cam0.map2, cv::INTER_LINEAR);
        orb_->detectAndCompute(f0_.img, cv::noArray(), f0_.kp, f0_.desc);

        if (frame_cb_) {
            FrameData fd;
            fd.stamp     = latest_stamp_;
            fd.frame_idx = frame_idx_++;
            fd.cam0      = f0_.img;
            fd.stereo    = display_stereo_;
            fd.kp        = display_kp_;
            fd.depth     = display_depth_;
            fd.flow      = display_flow_;
            fd.n_tracked = tracked_pts3d_.size();
            fd.T_pnp     = T_world_cam_;
            frame_cb_(fd);
        } else {
            frame_idx_++;
        }

        tryProcess();
    }

    void onCam1(const BagMessage& msg) {
        cv::Mat raw = decodeImage(msg);
        if (raw.empty()) return;
        f1_.stamp = msg.stamp.toSec();
        cv::remap(raw, f1_.img, calib_.cam1.map1, calib_.cam1.map2, cv::INTER_LINEAR);
        tryProcess();
    }

    void tryProcess() {
        if (!f0_.ready() || !f1_.ready()) return;
        double dt = std::abs(f0_.stamp - f1_.stamp);
        if (dt > 0.04) {
            if (f0_.stamp < f1_.stamp) f0_.clear();
            else                       f1_.clear();
            return;
        }
        processPair();
        f0_.clear();
        f1_.clear();
    }

    // ── KLT + PnP ─────────────────────────────────────────────────────────────
    bool trackAndPose(const cv::Mat& curr_img) {
        if (prev_img_.empty() || tracked_pts2d_.empty()) return false;

        std::vector<cv::Point2f> curr_pts;
        std::vector<uchar>       status;
        cv::calcOpticalFlowPyrLK(prev_img_, curr_img,
                                  tracked_pts2d_, curr_pts,
                                  status, cv::noArray());

        std::vector<cv::Point3f> pts3d_ok;
        std::vector<cv::Point2f> pts2d_ok, pts2d_prev_ok;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                pts3d_ok.push_back(tracked_pts3d_[i]);
                pts2d_ok.push_back(curr_pts[i]);
                pts2d_prev_ok.push_back(tracked_pts2d_[i]);
            }
        }
        tracked_pts3d_ = pts3d_ok;
        tracked_pts2d_ = pts2d_ok;

        if ((int)pts3d_ok.size() < 6) return false;

        cv::Mat rvec, tvec, inliers;
        bool ok = cv::solvePnPRansac(pts3d_ok, pts2d_ok,
                                      calib_.cam0.K, cv::Mat(),
                                      rvec, tvec,
                                      false, 100, 2.0f, 0.99, inliers);
        if (!ok || inliers.rows < 6) return false;

        std::vector<bool>        is_inlier(pts3d_ok.size(), false);
        std::vector<cv::Point3f> pts3d_in;
        std::vector<cv::Point2f> pts2d_in;
        for (int i = 0; i < inliers.rows; i++) {
            int idx = inliers.at<int>(i, 0);
            is_inlier[idx] = true;
            pts3d_in.push_back(pts3d_ok[idx]);
            pts2d_in.push_back(pts2d_ok[idx]);
        }
        tracked_pts3d_ = pts3d_in;
        tracked_pts2d_ = pts2d_in;

        display_flow_.clear();
        for (size_t i = 0; i < pts2d_ok.size(); i++)
            display_flow_.push_back({pts2d_prev_ok[i], pts2d_ok[i], is_inlier[i]});

        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d Re;
        Eigen::Vector3d te;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                Re(r, c) = R.at<double>(r, c);
        te << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
        T_world_cam_ = Sophus::SE3d(Re, te).inverse();
        return true;
    }

    void seedLandmarks(const std::vector<cv::DMatch>& matches, const cv::Mat& pts4D) {
        for (int i = 0; i < (int)matches.size(); i++) {
            float W = pts4D.at<float>(3, i);
            if (std::abs(W) < 1e-9f) continue;
            float Z = pts4D.at<float>(2, i) / W;
            if (Z <= 0.f || Z > 10.f) continue;
            float X = pts4D.at<float>(0, i) / W;
            float Y = pts4D.at<float>(1, i) / W;
            Eigen::Vector3d pt_world = T_world_cam_ * Eigen::Vector3d(X, Y, Z);
            tracked_pts3d_.push_back({(float)pt_world.x(),
                                       (float)pt_world.y(),
                                       (float)pt_world.z()});
            tracked_pts2d_.push_back(f0_.kp[matches[i].queryIdx].pt);
        }
    }

    // ── Full stereo pipeline ──────────────────────────────────────────────────
    void processPair() {
        // Track and estimate pose; fire callback so main can update EKF
        if (trackAndPose(f0_.img) && pose_cb_)
            pose_cb_(latest_stamp_, T_world_cam_, tracked_pts3d_.size());

        // Detect and triangulate stereo pair
        std::vector<cv::KeyPoint> kp1;
        cv::Mat desc1;
        orb_->detectAndCompute(f1_.img, cv::noArray(), kp1, desc1);

        std::vector<cv::DMatch> matches;
        if (!f0_.desc.empty() && !desc1.empty()) {
            cv::BFMatcher matcher(cv::NORM_HAMMING, /*crossCheck=*/true);
            matcher.match(f0_.desc, desc1, matches);
            std::sort(matches.begin(), matches.end(),
                [](auto& a, auto& b){ return a.distance < b.distance; });
            if (matches.size() > 150) matches.resize(150);
        }

        cv::Mat pts4D;
        if (!matches.empty()) {
            std::vector<cv::Point2f> pts0, pts1;
            for (auto& m : matches) {
                pts0.push_back(f0_.kp[m.queryIdx].pt);
                pts1.push_back(kp1[m.trainIdx].pt);
            }
            cv::triangulatePoints(P0_, P1_, pts0, pts1, pts4D);
        }

        if ((int)tracked_pts3d_.size() < kMinTrackPts && !pts4D.empty()) {
            tracked_pts3d_.clear();
            tracked_pts2d_.clear();
            seedLandmarks(matches, pts4D);
        }

        prev_img_ = f0_.img.clone();

        // Update display snapshot for the next frame callback
        display_depth_.clear();
        if (!pts4D.empty()) {
            for (int i = 0; i < (int)matches.size(); i++) {
                float W = pts4D.at<float>(3, i);
                if (std::abs(W) < 1e-9f) continue;
                float Z = pts4D.at<float>(2, i) / W;
                if (Z > 0.f && Z < 10.f)
                    display_depth_[matches[i].queryIdx] = Z;
            }
        }
        display_kp_ = f0_.kp;

        cv::drawMatches(f0_.img, f0_.kp, f1_.img, kp1, matches, display_stereo_,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                        cv::DrawMatchesFlags::DEFAULT);
    }
};
