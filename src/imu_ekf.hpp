#pragma once

#include "calibration.hpp"
#include <Eigen/Dense>
#include <sophus/se3.hpp>

// ── Error-State Extended Kalman Filter (ESKF) ─────────────────────────────────
//
// Nominal state (on manifold):
//   p  ∈ R³       position in world frame
//   v  ∈ R³       velocity in world frame
//   R  ∈ SO(3)    orientation: rotates body vectors into world frame
//   ba ∈ R³       accelerometer bias
//   bg ∈ R³       gyroscope bias
//
// Error state δx ∈ R¹⁵ (flat, lives in the tangent space around nominal):
//   [δp(0:3), δv(3:6), δθ(6:9), δba(9:12), δbg(12:15)]
//   δθ is a rotation vector — the SO(3) error: R_true = R_nominal · Exp(δθ)
//
// Why a separate error state?
//   Quaternion/SO3 can't be added directly — you can't do R += something.
//   The error state lives in flat R¹⁵, so normal linear algebra applies.
//   After each update the error is injected into the nominal state and reset to 0.
//
// ── PROPAGATION (IMU, ~200 Hz) ────────────────────────────────────────────────
//
//   Each IMU sample (a_m, ω_m) advances the nominal state forward by dt:
//
//     a_body = a_m - ba              correct for bias
//     ω_body = ω_m - bg
//
//     p  ← p + v·dt + ½·R·a_body·dt² + ½·g·dt²   position
//     v  ← v + R·a_body·dt + g·dt                 velocity
//     R  ← R · Exp(ω_body·dt)                     orientation (right-multiply)
//
//   The covariance is propagated via the linearised dynamics:
//
//     P ← F·P·Fᵀ + Q
//
//   F (15×15) is the discrete state-transition Jacobian — it says "if the error
//   state is δx now, where does it end up after dt?"  The key non-trivial blocks:
//     F[δv, δθ]  = -R · skew(a_body) · dt   (misalignment error rotates accel)
//     F[δv, δba] = -R · dt                  (bias error propagates into velocity)
//     F[δθ, δθ]  = I - skew(ω_body) · dt   (rotation error drifts with gyro)
//     F[δθ, δbg] = -I · dt                  (gyro bias error accumulates in angle)
//
//   Q (15×15) is the process noise: IMU white noise (na, ng) and bias random
//   walks (nba, nbg) injected directly into the velocity and rotation error states.
//
// ── UPDATE (PnP pose, ~20 Hz) ─────────────────────────────────────────────────
//
//   PnP gives an absolute pose T_world_cam = (p_meas, R_meas).
//   The measurement function is just h(x) = [p, R], so:
//
//     innovation_p = p_meas - p_nominal              3D position error
//     innovation_R = Log(R_nominal⁻¹ · R_meas)      rotation error as vector
//
//   H (6×15) maps the error state to the measurement space — trivially identity
//   in the position and rotation blocks (the camera directly observes both).
//
//   Standard Kalman update:
//     S  = H·P·Hᵀ + V           innovation covariance
//     K  = P·Hᵀ·S⁻¹             Kalman gain (15×6)
//     δx = K · [innov_p; innov_R]
//     P  = (I−KH)·P·(I−KH)ᵀ + K·V·Kᵀ   (Joseph form, numerically stable)
//
//   K decides how much to trust the camera vs. the IMU: large P (uncertain IMU)
//   → large K → lean on camera. Large V (uncertain camera) → small K → lean on IMU.
//
//   The 15D correction δx is then injected back into the nominal state:
//     p  ← p + δp
//     v  ← v + δv
//     R  ← R · Exp(δθ)          right-multiply on SO(3)
//     ba ← ba + δba
//     bg ← bg + δbg
//
// Measurement noise for PnP pose update — tune to match your PnP quality
struct ImuEkfUpdateNoise {
    double sigma_p = 0.05;   // position uncertainty [m]
    double sigma_R = 0.05;   // rotation uncertainty [rad] (~3°)
};

class ImuEkf {
public:
    explicit ImuEkf(const ImuCalib& imu,
                    ImuEkfUpdateNoise vnoise = ImuEkfUpdateNoise{},
                    Eigen::Vector3d gravity = Eigen::Vector3d{0, 0, -9.81})
        : na_(imu.accel_noise_density)
        , ng_(imu.gyro_noise_density)
        , nba_(imu.accel_random_walk)
        , nbg_(imu.gyro_random_walk)
        , vnoise_(vnoise)
        , gravity_(gravity)
    {
        p_  = Eigen::Vector3d::Zero();
        v_  = Eigen::Vector3d::Zero();
        R_  = Sophus::SO3d();
        ba_ = Eigen::Vector3d::Zero();
        bg_ = Eigen::Vector3d::Zero();

        P_  = Matrix15d::Identity() * 1e-4;
    }

    // ── Feed one IMU sample — call at 200 Hz ──────────────────────────────────
    void propagate(double stamp, const Eigen::Vector3d& a_m, const Eigen::Vector3d& w_m) {
        if (last_stamp_ < 0.0) { last_stamp_ = stamp; return; }

        double dt = stamp - last_stamp_;
        last_stamp_ = stamp;
        if (dt <= 0.0 || dt > 0.5) return;

        // Bias-corrected measurements
        const Eigen::Vector3d a = a_m - ba_;
        const Eigen::Vector3d w = w_m - bg_;

        // ── Nominal state integration (Euler) ─────────────────────────────────
        const Eigen::Vector3d a_world = R_ * a + gravity_;
        p_ += v_ * dt + 0.5 * a_world * dt * dt;
        v_ += a_world * dt;
        R_  = R_ * Sophus::SO3d::exp(w * dt);

        // ── Error-state covariance propagation: P = F·P·Fᵀ + Q ───────────────
        Matrix15d F = Matrix15d::Identity();

        // δp += δv · dt
        F.block<3,3>(0, 3)  = Eigen::Matrix3d::Identity() * dt;
        // δv  rotated-accel cross-coupling and bias injection
        F.block<3,3>(3, 6)  = -R_.matrix() * skew(a) * dt;
        F.block<3,3>(3, 9)  = -R_.matrix() * dt;
        // δθ gyro drift and bias injection
        F.block<3,3>(6, 6)  = Eigen::Matrix3d::Identity() - skew(w) * dt;
        F.block<3,3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;

        // Discrete process noise — noise densities are per √Hz, so scale by dt
        Matrix15d Q = Matrix15d::Zero();
        Q.block<3,3>(3,  3)  = Eigen::Matrix3d::Identity() * (na_  * na_  * dt);
        Q.block<3,3>(6,  6)  = Eigen::Matrix3d::Identity() * (ng_  * ng_  * dt);
        Q.block<3,3>(9,  9)  = Eigen::Matrix3d::Identity() * (nba_ * nba_ * dt);
        Q.block<3,3>(12, 12) = Eigen::Matrix3d::Identity() * (nbg_ * nbg_ * dt);

        P_ = F * P_ * F.transpose() + Q;
    }

    // ── Feed PnP absolute pose — call at ~20 Hz ───────────────────────────────
    void update(const Sophus::SE3d& T_world_cam) {
        const Eigen::Vector3d p_meas = T_world_cam.translation();
        const Sophus::SO3d    R_meas = T_world_cam.so3();

        // Innovation
        Eigen::Matrix<double, 6, 1> innov;
        innov.head<3>() = p_meas - p_;
        innov.tail<3>() = (R_.inverse() * R_meas).log();

        // H (6×15): camera directly observes position (block 0) and rotation (block 6)
        Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
        H.block<3,3>(0, 0) = Eigen::Matrix3d::Identity();
        H.block<3,3>(3, 6) = Eigen::Matrix3d::Identity();

        // Measurement noise
        Eigen::Matrix<double, 6, 6> V = Eigen::Matrix<double, 6, 6>::Zero();
        V.block<3,3>(0, 0) = Eigen::Matrix3d::Identity() * (vnoise_.sigma_p * vnoise_.sigma_p);
        V.block<3,3>(3, 3) = Eigen::Matrix3d::Identity() * (vnoise_.sigma_R * vnoise_.sigma_R);

        // Kalman gain
        const Eigen::Matrix<double, 6, 6>  S = H * P_ * H.transpose() + V;
        const Eigen::Matrix<double, 15, 6> K = P_ * H.transpose() * S.inverse();

        // Error state
        const Eigen::Matrix<double, 15, 1> dx = K * innov;

        // Inject correction into nominal state
        p_  += dx.segment<3>(0);
        v_  += dx.segment<3>(3);
        R_   = R_ * Sophus::SO3d::exp(dx.segment<3>(6));
        ba_ += dx.segment<3>(9);
        bg_ += dx.segment<3>(12);

        // Covariance update — Joseph form: (I−KH)·P·(I−KH)ᵀ + K·V·Kᵀ
        const Matrix15d IKH = Matrix15d::Identity() - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * V * K.transpose();
    }

    Sophus::SE3d    pose()       const { return {R_, p_}; }
    Eigen::Vector3d velocity()   const { return v_; }
    Eigen::Vector3d accelBias()  const { return ba_; }
    Eigen::Vector3d gyroBias()   const { return bg_; }
    bool            initialised() const { return last_stamp_ >= 0.0; }

private:
    using Matrix15d = Eigen::Matrix<double, 15, 15>;

    static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S <<     0, -v.z(),  v.y(),
             v.z(),      0, -v.x(),
            -v.y(),  v.x(),      0;
        return S;
    }

    // IMU noise parameters (from calibration)
    double na_, ng_, nba_, nbg_;
    ImuEkfUpdateNoise vnoise_;
    Eigen::Vector3d gravity_;

    double last_stamp_ = -1.0;

    // Nominal state
    Eigen::Vector3d p_;
    Eigen::Vector3d v_;
    Sophus::SO3d    R_;
    Eigen::Vector3d ba_;
    Eigen::Vector3d bg_;

    // Error-state covariance
    Matrix15d P_;
};
