#pragma once
#include "calibration.hpp"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <deque>
#include <map>
#include <fstream>
#include <iomanip>

// ── IMU preintegration ────────────────────────────────────────────────────────
// Collapses all IMU samples between two keyframes into {dR, dv, dp} expressed
// in the body frame at the start of the interval, plus a 9×9 noise covariance.
struct PreintegratedImu {
    Sophus::SO3d              dR;
    Eigen::Vector3d           dv{0,0,0}, dp{0,0,0};
    double                    dt = 0;
    Eigen::Matrix<double,9,9> cov = Eigen::Matrix<double,9,9>::Zero();

    void reset() { *this = {}; }

    void integrate(double step,
                   const Eigen::Vector3d& am, const Eigen::Vector3d& wm,
                   const Eigen::Vector3d& ba, const Eigen::Vector3d& bg,
                   double na, double ng)
    {
        Eigen::Vector3d a = am - ba, w = wm - bg;
        Eigen::Vector3d aw = dR * a;
        dp  += dv * step + 0.5 * aw * step * step;
        dv  += aw * step;
        dR   = dR * Sophus::SO3d::exp(w * step);
        dt  += step;
        cov.block<3,3>(0,0) += Eigen::Matrix3d::Identity() * (ng*ng*step);
        cov.block<3,3>(3,3) += dR.matrix() * (na*na*step) * dR.matrix().transpose();
    }
};

// ── IMU factor  (15D residual) ────────────────────────────────────────────────
// Connects two consecutive keyframes via their preintegrated IMU interval.
// Residuals: r_dp, r_dv, r_dR (whitened by sqrt-info), r_dba, r_dbg.
struct ImuFactor {
    ImuFactor(const PreintegratedImu& p, Eigen::Vector3d g = {0,0,-9.81})
        : pim(p), g(std::move(g))
        , W(p.cov.inverse().llt().matrixL().transpose()) {}

    template<typename T>
    bool operator()(const T* ti, const T* qi, const T* vi, const T* bi,
                    const T* tj, const T* qj, const T* vj, const T* bj,
                    T* r) const
    {
        using V3 = Eigen::Matrix<T,3,1>;
        Eigen::Quaternion<T> qi_(qi[3],qi[0],qi[1],qi[2]);
        Eigen::Quaternion<T> qj_(qj[3],qj[0],qj[1],qj[2]);
        auto Ri = qi_.normalized().toRotationMatrix();
        auto Rj = qj_.normalized().toRotationMatrix();
        V3 pi(ti[0],ti[1],ti[2]), pj(tj[0],tj[1],tj[2]);
        V3 vi_(vi[0],vi[1],vi[2]), vj_(vj[0],vj[1],vj[2]);
        T  d = T(pim.dt);
        V3 gv = g.cast<T>();

        V3 r_dp = Ri.transpose()*(pj-pi-vi_*d-T(0.5)*gv*d*d) - pim.dp.cast<T>();
        V3 r_dv = Ri.transpose()*(vj_-vi_-gv*d)               - pim.dv.cast<T>();

        auto dR_e = pim.dR.matrix().cast<T>().transpose() * Ri.transpose() * Rj;
        T tr = ceres::fmax(T(-1), ceres::fmin(T(1), (dR_e.trace()-T(1))/T(2)));
        T angle = ceres::acos(tr);
        V3 r_dR = ceres::abs(angle) < T(1e-8) ? V3::Zero()
            : angle / (T(2)*ceres::sin(angle))
              * V3(dR_e(2,1)-dR_e(1,2), dR_e(0,2)-dR_e(2,0), dR_e(1,0)-dR_e(0,1));

        Eigen::Map<Eigen::Matrix<T,15,1>> res(r);
        res << r_dp, r_dv, r_dR,
               V3(bj[0]-bi[0], bj[1]-bi[1], bj[2]-bi[2]),
               V3(bj[3]-bi[3], bj[4]-bi[4], bj[5]-bi[5]);
        res.template head<9>() = W.cast<T>() * res.template head<9>();
        return true;
    }

    static ceres::CostFunction* create(const PreintegratedImu& p) {
        return new ceres::AutoDiffCostFunction<ImuFactor,15, 3,4,3,6, 3,4,3,6>(new ImuFactor(p));
    }

    PreintegratedImu          pim;
    Eigen::Vector3d           g;
    Eigen::Matrix<double,9,9> W;  // sqrt information
};

// ── Reprojection factor  (2D residual) ───────────────────────────────────────
// r = π(R_cam_world · (X_world − t_world_cam)) − x_obs
struct ReprojectionFactor {
    ReprojectionFactor(Eigen::Vector2d obs, double fx, double fy,
                       double cx, double cy, double sigma_px = 1.5)
        : obs(std::move(obs)), fx(fx), fy(fy), cx(cx), cy(cy), w(1.0/sigma_px) {}

    template<typename T>
    bool operator()(const T* tc, const T* qc, const T* Xw, T* r) const {
        using V3 = Eigen::Matrix<T,3,1>;
        Eigen::Quaternion<T> q(qc[3],qc[0],qc[1],qc[2]);
        V3 Xc = q.normalized().toRotationMatrix().transpose()
                * (V3(Xw[0],Xw[1],Xw[2]) - V3(tc[0],tc[1],tc[2]));
        if (Xc.z() < T(1e-4)) { r[0] = r[1] = T(0); return true; }
        r[0] = T(w) * (T(fx)*Xc.x()/Xc.z() + T(cx) - T(obs.x()));
        r[1] = T(w) * (T(fy)*Xc.y()/Xc.z() + T(cy) - T(obs.y()));
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector2d& obs, const cv::Mat& K,
                                        double sigma_px = 1.5) {
        return new ceres::AutoDiffCostFunction<ReprojectionFactor,2, 3,4,3>(
            new ReprojectionFactor(obs, K.at<double>(0,0), K.at<double>(1,1),
                                        K.at<double>(0,2), K.at<double>(1,2), sigma_px));
    }

    Eigen::Vector2d obs;
    double fx, fy, cx, cy, w;
};

// ── Keyframe node — raw double arrays that Ceres optimises ───────────────────
struct Keyframe {
    double t[3]{};
    double q[4]{0,0,0,1};   // [qx,qy,qz,qw] — Eigen memory order
    double vel[3]{};
    double bias[6]{};        // [ba(3), bg(3)]
    double stamp = 0;

    void set(const Sophus::SE3d& T, const Eigen::Vector3d& v = {}) {
        auto eq = T.unit_quaternion();
        Eigen::Map<Eigen::Vector3d>(t)   = T.translation();
        q[0]=eq.x(); q[1]=eq.y(); q[2]=eq.z(); q[3]=eq.w();
        Eigen::Map<Eigen::Vector3d>(vel) = v;
    }

    Sophus::SE3d    pose()     const {
        return {Eigen::Quaterniond(q[3],q[0],q[1],q[2]).normalized(),
                Eigen::Map<const Eigen::Vector3d>(t)};
    }
    Eigen::Vector3d velocity() const { return Eigen::Map<const Eigen::Vector3d>(vel); }
    Eigen::Vector3d ba()       const { return Eigen::Map<const Eigen::Vector3d>(bias); }
    Eigen::Vector3d bg()       const { return Eigen::Map<const Eigen::Vector3d>(bias+3); }
};

// ── Sliding-window factor graph ───────────────────────────────────────────────
class SlidingWindowGraph {
public:
    static constexpr int kWindow = 10;

    explicit SlidingWindowGraph(const StereoCalib& c) : calib_(c) {}

    void addKeyframe(const Sophus::SE3d& pose, const Eigen::Vector3d& vel,
                     const PreintegratedImu& pim, double stamp)
    {
        auto kf = std::make_shared<Keyframe>();
        kf->stamp = stamp;
        kf->set(pose, vel);
        if (!kfs_.empty()) {
            Eigen::Map<Eigen::Vector3d>(kf->bias)   = kfs_.back()->ba();
            Eigen::Map<Eigen::Vector3d>(kf->bias+3) = kfs_.back()->bg();
        }

        problem_.AddParameterBlock(kf->t,    3);
        problem_.AddParameterBlock(kf->q,    4, new ceres::EigenQuaternionManifold);
        problem_.AddParameterBlock(kf->vel,  3);
        problem_.AddParameterBlock(kf->bias, 6);

        if (kfs_.empty()) {
            // Ceres built-in: freeze first frame to fix gauge freedom
            problem_.SetParameterBlockConstant(kf->t);
            problem_.SetParameterBlockConstant(kf->q);
            problem_.SetParameterBlockConstant(kf->vel);
        } else {
            problem_.AddResidualBlock(
                ImuFactor::create(pim), new ceres::HuberLoss(0.5),
                kfs_.back()->t, kfs_.back()->q, kfs_.back()->vel, kfs_.back()->bias,
                kf->t,          kf->q,          kf->vel,          kf->bias);
        }

        kfs_.push_back(kf);
        if ((int)kfs_.size() > kWindow) drop();
    }

    void addReprojection(int kf_idx, int lm_id,
                          const Eigen::Vector3d& Xw, const Eigen::Vector2d& obs)
    {
        if (!lms_.count(lm_id)) {
            lms_[lm_id] = Xw;
            problem_.AddParameterBlock(lms_[lm_id].data(), 3);
        }
        auto& kf = kfs_[kf_idx];
        problem_.AddResidualBlock(
            ReprojectionFactor::create(obs, calib_.cam0.K),
            new ceres::HuberLoss(1.0),
            kf->t, kf->q, lms_[lm_id].data());
    }

    void solve() {
        ceres::Solver::Options o;
        o.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        o.max_num_iterations = 10;
        o.num_threads        = 4;
        ceres::Solver::Summary s;
        ceres::Solve(o, &problem_, &s);
    }

    // Append latest keyframe to a TUM-format trajectory file.
    // Visualise offline with:  evo_traj tum trajectory.txt --plot
    //                    or:   python3 -c "
    //                            import numpy as np, matplotlib.pyplot as plt
    //                            d = np.loadtxt('trajectory.txt')
    //                            plt.plot(d[:,1], d[:,3]); plt.axis('equal'); plt.show()"
    void writeTUM(const std::string& path) const {
        if (kfs_.empty()) return;
        std::ofstream f(path, std::ios::app);
        auto& kf = *kfs_.back();
        auto  t  = kf.pose().translation();
        auto  q  = kf.pose().unit_quaternion();
        f << std::fixed << std::setprecision(9)
          << kf.stamp << " "
          << t.x() << " " << t.y() << " " << t.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }

    Sophus::SE3d latestPose() const { return kfs_.back()->pose(); }

private:
    // TODO: replace with proper Schur-complement marginalisation.
    // Currently just drops the oldest frame, introducing a small discontinuity.
    void drop() {
        auto& f = kfs_.front();
        problem_.RemoveParameterBlock(f->t);
        problem_.RemoveParameterBlock(f->q);
        problem_.RemoveParameterBlock(f->vel);
        problem_.RemoveParameterBlock(f->bias);
        kfs_.pop_front();
    }

    const StereoCalib&                    calib_;
    ceres::Problem                        problem_;
    std::deque<std::shared_ptr<Keyframe>> kfs_;
    std::map<int, Eigen::Vector3d>        lms_;
};
