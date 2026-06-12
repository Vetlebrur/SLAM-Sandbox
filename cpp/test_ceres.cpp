#include <ceres/ceres.h>
#include <iostream>
#include <random>

// Fit y = a * exp(b * x) + c to noisy observations.
struct ExpResidual {
    ExpResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const p, T* r) const {
        r[0] = T(y_) - (p[0] * ceres::exp(p[1] * T(x_)) + p[2]);
        return true;
    }

    static ceres::CostFunction* Create(double x, double y) {
        return new ceres::AutoDiffCostFunction<ExpResidual, 1, 3>(
            new ExpResidual(x, y));
    }

    double x_, y_;
};

int main() {
    // Ground truth: y = 3 * exp(0.4 * x) - 1
    const double a_true = 3.0, b_true = 0.4, c_true = -1.0;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.2);

    ceres::Problem problem;
    for (int i = 0; i < 25; ++i) {
        double x = i * 0.3;
        double y = a_true * std::exp(b_true * x) + c_true + noise(rng);
        problem.AddResidualBlock(ExpResidual::Create(x, y), nullptr, new double[3]{1.0, 0.0, 0.0});
    }

    // Restart with shared parameter block
    problem = ceres::Problem{};
    double params[3] = {1.0, 0.0, 0.0};
    for (int i = 0; i < 25; ++i) {
        double x = i * 0.3;
        double y = a_true * std::exp(b_true * x) + c_true + noise(rng);
        problem.AddResidualBlock(ExpResidual::Create(x, y), nullptr, params);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type         = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    std::cout << "Fitting  y = a * exp(b*x) + c\n\n";
    std::printf("  True:   a=%.4f  b=%.4f  c=%.4f\n", a_true, b_true, c_true);
    std::printf("  Solved: a=%.4f  b=%.4f  c=%.4f\n\n", params[0], params[1], params[2]);
    std::cout << summary.BriefReport() << "\n";
    return 0;
}
