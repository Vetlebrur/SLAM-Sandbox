#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // ── SO3: rotation around Z ────────────────────────────────────────────
    Sophus::SO3d R = Sophus::SO3d::rotZ(M_PI / 2.0);
    std::cout << "SO3 (90 deg around Z):\n" << R.matrix() << "\n\n";

    // Lie algebra: log map gives axis-angle vector
    Eigen::Vector3d omega = R.log();
    std::cout << "log(R) axis-angle: " << omega.transpose()
              << "   |omega| = " << omega.norm()
              << "  (should be pi/2 = " << M_PI / 2.0 << ")\n\n";

    // ── SE3: rigid transform = rotation + translation ────────────────────
    Sophus::SE3d T(R, Eigen::Vector3d(1.0, 2.0, 0.0));

    Eigen::Vector3d p(1.0, 0.0, 0.0);
    std::cout << "SE3 * (1,0,0) = " << (T * p).transpose() << "\n";
    std::cout << "T.inv * T * p = " << (T.inverse() * T * p).transpose()
              << "  (should be p)\n\n";

    // ── Perturbation via Lie algebra (key for SLAM optimisation) ─────────
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.01, 0.0, 0.0,   // translation perturbation
          0.0,  0.0, 0.05;  // rotation perturbation (small angle around Z)
    Sophus::SE3d T2 = Sophus::SE3d::exp(xi) * T;
    std::cout << "Perturbed translation: " << T2.translation().transpose() << "\n";
    std::cout << "log(T2) tangent vec:   " << T2.log().transpose() << "\n\n";

    // ── Compose chain of poses ────────────────────────────────────────────
    std::vector<Sophus::SE3d> poses;
    poses.push_back(Sophus::SE3d::trans(1.0, 0.0, 0.0));
    poses.push_back(Sophus::SE3d(Sophus::SO3d::rotZ(M_PI / 4.0), Eigen::Vector3d(0, 1, 0)));
    poses.push_back(Sophus::SE3d::trans(0.5, 0.5, 0.0));

    Sophus::SE3d T_chain = Sophus::SE3d::trans(0, 0, 0);
    for (auto& Ti : poses) T_chain = Ti * T_chain;
    std::cout << "Chained pose translation: " << T_chain.translation().transpose() << "\n";

    return 0;
}
