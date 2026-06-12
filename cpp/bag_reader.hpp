#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct RosTime {
    uint32_t sec  = 0;
    uint32_t nsec = 0;
    double toSec() const { return sec + nsec * 1e-9; }
};

struct BagMessage {
    std::string             topic;
    std::string             type;   // e.g. "sensor_msgs/Imu"
    RosTime                 stamp;
    std::vector<uint8_t>    data;   // serialised ROS message payload
};

using MessageCallback = std::function<void(const BagMessage&)>;

// Decode a sensor_msgs/Image payload into a cv::Mat.
// Returns an empty Mat if the encoding is unsupported.
cv::Mat decodeImage(const BagMessage& msg);

// Minimal ROS1 bag V2.0 reader (no ROS dependency).
// Supports compression: none, lz4.
class BagReader {
public:
    explicit BagReader(const std::string& path);

    // Iterate over every message in the bag. If `topics` is non-empty only
    // messages on those topics are delivered.
    void readMessages(const MessageCallback& cb,
                      const std::vector<std::string>& topics = {});

private:
    std::string path_;

    struct Connection { std::string topic, type; };
    std::map<uint32_t, Connection> conns_;

    using Header = std::map<std::string, std::vector<uint8_t>>;
    static Header parseHeader(const uint8_t* buf, uint32_t len);

    void processChunk(const uint8_t* data, size_t data_size,
                      const std::string& compression,
                      uint32_t uncompressed_size,
                      const MessageCallback& cb,
                      const std::set<std::string>& filter);

    void dispatchRecord(const uint8_t* hdr_buf, uint32_t hdr_len,
                        const uint8_t* data_buf, uint32_t data_len,
                        const MessageCallback& cb,
                        const std::set<std::string>& filter);
};
