#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

static constexpr int W = 1200, H = 800;

// ── Camera ─────────────────────────────────────────────────────────────────
struct Camera {
    float az   = 0.f;
    float el   = 0.8f;   // elevation in radians (0=horizon, pi/2=top-down)
    float dist = 10.f;
    Eigen::Vector3f target{0.f, 0.f, 0.f};
};

static Eigen::Matrix4f viewMat(const Camera& c) {
    float ce = cosf(c.el), se = sinf(c.el);
    float ca = cosf(c.az), sa = sinf(c.az);
    Eigen::Vector3f eye = c.target + c.dist * Eigen::Vector3f(ce * sa, -ce * ca, se);

    Eigen::Vector3f up(0.f, 0.f, 1.f);
    if (std::abs(c.el) > 1.55f) up = {0.f, 1.f, 0.f};

    Eigen::Vector3f fwd = (c.target - eye).normalized();
    Eigen::Vector3f rgt = fwd.cross(up).normalized();
    Eigen::Vector3f u   = rgt.cross(fwd);

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V(0,0) = rgt.x(); V(0,1) = rgt.y(); V(0,2) = rgt.z(); V(0,3) = -rgt.dot(eye);
    V(1,0) = u.x();   V(1,1) = u.y();   V(1,2) = u.z();   V(1,3) = -u.dot(eye);
    V(2,0) = -fwd.x(); V(2,1) = -fwd.y(); V(2,2) = -fwd.z(); V(2,3) = fwd.dot(eye);
    return V;
}

static Eigen::Matrix4f projMat() {
    constexpr float fov = 60.f * (float)M_PI / 180.f;
    constexpr float n = 0.05f, f = 300.f;
    const float     s = 1.f / tanf(fov * 0.5f);
    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0,0) = s * H / W;
    P(1,1) = s;
    P(2,2) = (f + n) / (n - f);
    P(2,3) = 2.f * f * n / (n - f);
    P(3,2) = -1.f;
    return P;
}

// ── Height colormap: blue → cyan → green → yellow → red (BGR) ──────────────
static cv::Vec3b heightColor(float t) {
    t = std::clamp(t * 4.f, 0.f, 3.9999f);
    int   seg = (int)t;
    float s   = t - seg;
    auto  L   = [](float a, float b, float t) -> uint8_t { return (uint8_t)(a + (b - a) * t); };
    switch (seg) {
        case 0: return { 255,          L(0, 255, s), 0 };           // blue → cyan
        case 1: return { L(255, 0, s), 255,          0 };           // cyan → green
        case 2: return { 0,            255,          L(0, 255, s) }; // green → yellow
        default: return { 0,           L(255, 0, s), 255 };         // yellow → red
    }
}

// ── PLY loader ──────────────────────────────────────────────────────────────
struct Pt { float x, y, z; };

static std::vector<Pt> loadPly(const std::string& path, int subsample = 5) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    while (std::getline(f, line) && line != "end_header") {}

    std::vector<Pt> pts;
    pts.reserve(700000);
    int n = 0;
    float x, y, z, intensity;
    int r, g, b;
    while (f >> x >> y >> z >> intensity >> r >> g >> b) {
        if (++n % subsample == 0)
            pts.push_back({x, y, z});
    }
    return pts;
}

// ── Mouse state ─────────────────────────────────────────────────────────────
struct Mouse {
    bool  drag = false;
    bool  dirty = true;
    int   lx = 0, ly = 0;
    float daz = 0, del = 0, dzoom = 0;
};

static void onMouse(int event, int x, int y, int flags, void* ud) {
    auto& m = *static_cast<Mouse*>(ud);
    if (event == cv::EVENT_LBUTTONDOWN) {
        m.drag = true; m.lx = x; m.ly = y;
    } else if (event == cv::EVENT_LBUTTONUP) {
        m.drag = false;
    } else if (event == cv::EVENT_MOUSEMOVE && m.drag) {
        m.daz  += (x - m.lx) * 0.007f;
        m.del  += (y - m.ly) * 0.007f;
        m.lx = x; m.ly = y;
        m.dirty = true;
    } else if (event == cv::EVENT_MOUSEWHEEL) {
        m.dzoom -= cv::getMouseWheelDelta(flags) / 1200.f;
        m.dirty = true;
    }
}

// ── Render ───────────────────────────────────────────────────────────────────
static cv::Mat render(const std::vector<Pt>& pts,
                      const std::vector<Eigen::Vector3f>& traj,
                      const Camera& cam,
                      float zmin, float zrange) {
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(18, 18, 18));
    static std::vector<float> zbuf(W * H);
    std::fill(zbuf.begin(), zbuf.end(), 1.f);

    Eigen::Matrix4f mvp = projMat() * viewMat(cam);

    for (const auto& p : pts) {
        Eigen::Vector4f v = mvp * Eigen::Vector4f(p.x, p.y, p.z, 1.f);
        if (v.w() <= 0.f) continue;
        float iz = 1.f / v.w();
        float nx = v.x() * iz;
        float ny = v.y() * iz;
        float nz = v.z() * iz;
        if (nz < -1.f || nz > 1.f) continue;
        int px = (int)((nx + 1.f) * 0.5f * W);
        int py = (int)((1.f - (ny + 1.f) * 0.5f) * H);
        if ((unsigned)px >= (unsigned)W || (unsigned)py >= (unsigned)H) continue;
        int idx = py * W + px;
        if (nz >= zbuf[idx]) continue;
        zbuf[idx] = nz;
        img.at<cv::Vec3b>(py, px) = heightColor((p.z - zmin) / zrange);
    }

    // Trajectory: yellow line strip + dots
    if (traj.size() >= 2) {
        int ppx = -1, ppy = -1;
        for (const auto& tp : traj) {
            Eigen::Vector4f v = mvp * Eigen::Vector4f(tp.x(), tp.y(), tp.z(), 1.f);
            if (v.w() <= 0.f) { ppx = -1; continue; }
            float iz = 1.f / v.w();
            int px = (int)((v.x() * iz + 1.f) * 0.5f * W);
            int py = (int)((1.f - (v.y() * iz + 1.f) * 0.5f) * H);
            if ((unsigned)px >= (unsigned)W || (unsigned)py >= (unsigned)H) { ppx = -1; continue; }
            if (ppx >= 0)
                cv::line(img, {ppx, ppy}, {px, py}, {0, 220, 255}, 2, cv::LINE_AA);
            cv::circle(img, {px, py}, 4, {0, 180, 255}, -1, cv::LINE_AA);
            ppx = px; ppy = py;
        }
    }

    // HUD
    cv::putText(img, "Left-drag: orbit   Scroll: zoom   Q: quit",
                {10, H - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {180, 180, 180}, 1, cv::LINE_AA);
    return img;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    const std::string ply_path =
        "/home/vetlebrur/Projects/github/SLAM-Sandbox/vicon_room1/V1_01_easy"
        "/V1_01_easy/mav0/pointcloud0/data.ply";

    std::cout << "Loading point cloud (1-in-5 subsample)...\n" << std::flush;
    auto pts = loadPly(ply_path, 5);
    std::cout << "Loaded " << pts.size() << " points\n";

    // Centroid + height range
    float zmin = pts[0].z, zmax = pts[0].z;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& p : pts) {
        zmin = std::min(zmin, p.z);
        zmax = std::max(zmax, p.z);
        centroid += Eigen::Vector3f(p.x, p.y, p.z);
    }
    centroid /= (float)pts.size();
    float zrange = zmax - zmin;
    std::cout << "Height range: " << zmin << " to " << zmax << " m\n";

    Camera cam;
    cam.target = centroid;
    cam.dist   = zrange * 3.f;
    cam.el     = 0.9f;
    cam.az     = 0.f;

    // ── Add your SLAM trajectory here ────────────────────────────────────────
    // std::vector<Eigen::Vector3f> trajectory;
    // for (const auto& T : slam_poses)
    //     trajectory.push_back(T.translation().cast<float>());
    std::vector<Eigen::Vector3f> trajectory;

    Mouse ms;
    cv::namedWindow("pointcloud", cv::WINDOW_NORMAL);
    cv::resizeWindow("pointcloud", W, H);
    cv::setMouseCallback("pointcloud", onMouse, &ms);

    cv::Mat frame;
    while (true) {
        if (ms.dirty) {
            cam.az  += ms.daz;  ms.daz = 0;
            cam.el   = std::clamp(cam.el + ms.del, -1.5f, 1.5f); ms.del = 0;
            cam.dist = std::max(0.2f, cam.dist * std::exp(ms.dzoom)); ms.dzoom = 0;
            ms.dirty = false;
            frame = render(pts, trajectory, cam, zmin, zrange);
        }
        if (frame.empty())
            frame = render(pts, trajectory, cam, zmin, zrange);

        cv::imshow("pointcloud", frame);
        int key = cv::waitKey(30);
        if (key == 'q' || key == 27) break;
    }
    return 0;
}
