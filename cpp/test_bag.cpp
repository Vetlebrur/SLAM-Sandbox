#include "bag_reader.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

// ── ROS message deserialisation helpers ──────────────────────────────────────

static inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline double   rd_f64(const uint8_t* p) { double   v; memcpy(&v, p, 8); return v; }

// std_msgs/Header  →  returns bytes consumed
static size_t skip_header(const uint8_t* buf, size_t sz) {
    if (sz < 12) return sz;           // seq(4) + stamp(8)
    size_t pos = 12;
    if (pos + 4 > sz) return sz;
    uint32_t flen = rd_u32(buf + pos); pos += 4;
    return pos + flen;                // skip frame_id string
}

// sensor_msgs/Imu
struct ImuMsg {
    double stamp;
    double qx, qy, qz, qw;           // orientation
    double wx, wy, wz;               // angular_velocity
    double ax, ay, az;               // linear_acceleration
};

static bool decode_imu(const std::vector<uint8_t>& d, ImuMsg& out) {
    size_t pos = skip_header(d.data(), d.size());

    if (pos + 4 * 8 + 9 * 8 + 3 * 8 + 9 * 8 + 3 * 8 > d.size()) return false;

    // std_msgs/Header has stamp at offset 4 (after seq)
    out.stamp = rd_u32(d.data() + 4) + rd_u32(d.data() + 8) * 1e-9;

    // Quaternion (x, y, z, w)
    out.qx = rd_f64(d.data() + pos); pos += 8;
    out.qy = rd_f64(d.data() + pos); pos += 8;
    out.qz = rd_f64(d.data() + pos); pos += 8;
    out.qw = rd_f64(d.data() + pos); pos += 8;
    pos += 9 * 8; // orientation_covariance[9]

    // Angular velocity (x, y, z)
    out.wx = rd_f64(d.data() + pos); pos += 8;
    out.wy = rd_f64(d.data() + pos); pos += 8;
    out.wz = rd_f64(d.data() + pos); pos += 8;
    pos += 9 * 8; // angular_velocity_covariance[9]

    // Linear acceleration (x, y, z)
    out.ax = rd_f64(d.data() + pos);
    out.ay = rd_f64(d.data() + pos + 8);
    out.az = rd_f64(d.data() + pos + 16);
    return true;
}

// sensor_msgs/Image — just read the header fields we care about
struct ImageMsg {
    double   stamp;
    uint32_t height, width;
    std::string encoding;
};

static bool decode_image_header(const std::vector<uint8_t>& d, ImageMsg& out) {
    if (d.size() < 12) return false;
    out.stamp = rd_u32(d.data() + 4) + rd_u32(d.data() + 8) * 1e-9;
    size_t pos = skip_header(d.data(), d.size());
    if (pos + 8 > d.size()) return false;
    out.height = rd_u32(d.data() + pos); pos += 4;
    out.width  = rd_u32(d.data() + pos); pos += 4;
    if (pos + 4 > d.size()) return false;
    uint32_t enc_len = rd_u32(d.data() + pos); pos += 4;
    if (pos + enc_len > d.size()) return false;
    out.encoding = std::string(reinterpret_cast<const char*>(d.data() + pos), enc_len);
    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string bag_path = (argc > 1)
        ? argv[1]
        : "/home/vetlebrur/Projects/github/SLAM-Sandbox/vicon_room1/V1_01_easy/V1_01_easy.bag";

    std::cout << "Reading: " << bag_path << "\n\n";

    BagReader reader(bag_path);

    int imu_count = 0, img_count = 0;
    constexpr int PRINT_N = 5;

    reader.readMessages([&](const BagMessage& msg) {
        if (msg.topic == "/imu0") {
            ++imu_count;
            if (imu_count <= PRINT_N) {
                ImuMsg imu{};
                if (decode_imu(msg.data, imu)) {
                    std::cout << std::fixed << std::setprecision(6)
                              << "[IMU #" << imu_count
                              << "]  t=" << imu.stamp
                              << "  acc=(" << imu.ax << ", " << imu.ay << ", " << imu.az << ")"
                              << "  gyro=(" << imu.wx << ", " << imu.wy << ", " << imu.wz << ")\n";
                }
            }
        } else if (msg.topic == "/cam0/image_raw") {
            ++img_count;
            if (img_count == 1) {
                ImageMsg im{};
                if (decode_image_header(msg.data, im)) {
                    std::cout << "[Image #1]  t=" << std::fixed << std::setprecision(6)
                              << im.stamp << "  " << im.width << "x" << im.height
                              << "  enc=" << im.encoding << "\n";
                }
            }
        }
    }, {"/imu0", "/cam0/image_raw"});

    std::cout << "\n=== Summary ===\n"
              << "  /imu0           : " << imu_count << " messages\n"
              << "  /cam0/image_raw : " << img_count << " messages\n";
    return 0;
}
