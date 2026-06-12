#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <iostream>

int main() {
    std::cout << "Eigen version: "
              << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << "\n\n";

    // Rotation + translation
    Eigen::AngleAxisd aa(M_PI / 4.0, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d R = aa.toRotationMatrix();
    Eigen::Vector3d t(1.0, 2.0, 3.0);

    Eigen::Vector3d p(1.0, 0.0, 0.0);
    std::cout << "R (45 deg around Z):\n" << R << "\n\n";
    std::cout << "R*p + t = " << (R * p + t).transpose() << "\n\n";

    // Solve Ax = b
    Eigen::Matrix3d A;
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 10;
    Eigen::Vector3d b(3, 3, 4);
    Eigen::Vector3d x = A.lu().solve(b);
    std::cout << "Ax=b  x = " << x.transpose() << "\n";
    std::cout << "residual = " << (A * x - b).norm() << "\n\n";

    // SVD / pseudo-inverse demo
    Eigen::MatrixXd M(3, 2);
    M << 1, 2, 3, 4, 5, 6;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "Singular values of M:\n" << svd.singularValues().transpose() << "\n";

    return 0;
}
