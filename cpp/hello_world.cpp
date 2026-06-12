#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <iostream>

int main() {
    std::cout << "Hello, SLAM!\n\n";

    // Eigen
    Eigen::Vector3d p(1, 2, 3);
    std::cout << "Eigen vector: " << p.transpose() << "\n";

    // Sophus SE3
    Sophus::SE3d T(Sophus::SO3d::rotZ(M_PI / 4.0), Eigen::Vector3d(1, 0, 0));
    std::cout << "SE3 * p     : " << (T * p).transpose() << "\n";

    // Ceres version string
    std::cout << "Ceres       : " << CERES_VERSION_STRING << "\n";

    return 0;
}
