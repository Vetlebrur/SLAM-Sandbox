#include "bag_reader.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <lz4.h>

// ── ROS1 bag V2.0 record op codes ───────────────────────────────────────────
static constexpr uint8_t OP_MSG_DATA    = 0x02;
static constexpr uint8_t OP_FILE_HEADER = 0x03;
static constexpr uint8_t OP_INDEX_DATA  = 0x04;
static constexpr uint8_t OP_CHUNK      = 0x05;
static constexpr uint8_t OP_CHUNK_HDR  = 0x06;
static constexpr uint8_t OP_CONNECTION = 0x07;

// ── Little-endian helpers ────────────────────────────────────────────────────
static inline uint32_t u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t u64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

// ── Header parsing ───────────────────────────────────────────────────────────
BagReader::Header BagReader::parseHeader(const uint8_t* buf, uint32_t len) {
    Header fields;
    uint32_t pos = 0;
    while (pos + 4 <= len) {
        uint32_t fl = u32(buf + pos); pos += 4;
        if (pos + fl > len) break;
        const char* s  = reinterpret_cast<const char*>(buf + pos);
        const char* eq = static_cast<const char*>(memchr(s, '=', fl));
        if (eq) {
            std::string key(s, eq);
            fields[key] = std::vector<uint8_t>(buf + pos + (eq - s) + 1,
                                               buf + pos + fl);
        }
        pos += fl;
    }
    return fields;
}

// ── Constructor ──────────────────────────────────────────────────────────────
BagReader::BagReader(const std::string& path) : path_(path) {}

// ── Main read loop ───────────────────────────────────────────────────────────
void BagReader::readMessages(const MessageCallback& cb,
                             const std::vector<std::string>& topics) {
    std::ifstream f(path_, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open bag: " + path_);

    // Check magic "#ROSBAG V2.0\n"
    char magic[13]{};
    f.read(magic, 13);
    if (std::string(magic, 12) != "#ROSBAG V2.0")
        throw std::runtime_error("Not a ROS bag V2.0 file: " + path_);

    std::set<std::string> filter(topics.begin(), topics.end());

    std::vector<uint8_t> hbuf, dbuf;
    while (f.good()) {
        uint32_t hl{}, dl{};
        if (!f.read(reinterpret_cast<char*>(&hl), 4)) break;
        hbuf.resize(hl);
        if (!f.read(reinterpret_cast<char*>(hbuf.data()), hl)) break;
        if (!f.read(reinterpret_cast<char*>(&dl), 4)) break;
        dbuf.resize(dl);
        if (!f.read(reinterpret_cast<char*>(dbuf.data()), dl)) break;

        dispatchRecord(hbuf.data(), hl, dbuf.data(), dl, cb, filter);
    }
}

// ── Per-record dispatch ──────────────────────────────────────────────────────
void BagReader::dispatchRecord(const uint8_t* hbuf, uint32_t hl,
                               const uint8_t* dbuf, uint32_t dl,
                               const MessageCallback& cb,
                               const std::set<std::string>& filter) {
    auto hdr = parseHeader(hbuf, hl);
    auto it  = hdr.find("op");
    if (it == hdr.end()) return;
    uint8_t op = it->second[0];

    if (op == OP_CONNECTION) {
        uint32_t id = u32(hdr["conn"].data());
        Connection& c = conns_[id];
        c.topic = std::string(hdr["topic"].begin(), hdr["topic"].end());
        auto dhdr = parseHeader(dbuf, dl);
        if (dhdr.count("type"))
            c.type = std::string(dhdr["type"].begin(), dhdr["type"].end());

    } else if (op == OP_CHUNK) {
        std::string compression(hdr["compression"].begin(),
                                hdr["compression"].end());
        uint32_t uncompressed_size = u32(hdr["size"].data());
        processChunk(dbuf, dl, compression, uncompressed_size, cb, filter);
    }
    // FILE_HEADER, INDEX_DATA, CHUNK_HDR are skipped — not needed for reading.
}

// ── Chunk decompression + inner record scan ───────────────────────────────────
void BagReader::processChunk(const uint8_t* data, size_t data_size,
                             const std::string& compression,
                             uint32_t uncompressed_size,
                             const MessageCallback& cb,
                             const std::set<std::string>& filter) {
    std::vector<uint8_t> buf;

    if (compression == "none") {
        buf.assign(data, data + data_size);

    } else if (compression == "lz4") {
        buf.resize(uncompressed_size);
        int n = LZ4_decompress_safe(reinterpret_cast<const char*>(data),
                                    reinterpret_cast<char*>(buf.data()),
                                    static_cast<int>(data_size),
                                    static_cast<int>(uncompressed_size));
        if (n != static_cast<int>(uncompressed_size))
            throw std::runtime_error("LZ4 decompression failed (got " +
                                     std::to_string(n) + ")");

    } else if (compression == "bz2") {
        throw std::runtime_error(
            "bz2-compressed bag — recompress with:\n"
            "  rosbag compress --lz4 <file.bag>");

    } else {
        throw std::runtime_error("Unknown compression: " + compression);
    }

    // Scan inner records (CONNECTION + MSG_DATA)
    size_t pos = 0;
    while (pos + 8 <= buf.size()) {
        uint32_t rhl = u32(buf.data() + pos); pos += 4;
        if (pos + rhl > buf.size()) break;
        const uint8_t* rhbuf = buf.data() + pos; pos += rhl;
        if (pos + 4 > buf.size()) break;
        uint32_t rdl = u32(buf.data() + pos); pos += 4;
        if (pos + rdl > buf.size()) break;
        const uint8_t* rdbuf = buf.data() + pos; pos += rdl;

        auto rhdr = parseHeader(rhbuf, rhl);
        auto oit  = rhdr.find("op");
        if (oit == rhdr.end()) continue;
        uint8_t rop = oit->second[0];

        if (rop == OP_CONNECTION) {
            // Connections may appear inside chunks before their first message.
            uint32_t id = u32(rhdr["conn"].data());
            Connection& c = conns_[id];
            c.topic = std::string(rhdr["topic"].begin(), rhdr["topic"].end());
            auto dhdr = parseHeader(rdbuf, rdl);
            if (dhdr.count("type"))
                c.type = std::string(dhdr["type"].begin(), dhdr["type"].end());

        } else if (rop == OP_MSG_DATA) {
            uint32_t id = u32(rhdr["conn"].data());
            auto cit = conns_.find(id);
            if (cit == conns_.end()) continue;

            const std::string& topic = cit->second.topic;
            if (!filter.empty() && !filter.count(topic)) continue;

            BagMessage msg;
            msg.topic = topic;
            msg.type  = cit->second.type;
            msg.stamp.sec  = u32(rhdr["time"].data());
            msg.stamp.nsec = u32(rhdr["time"].data() + 4);
            msg.data.assign(rdbuf, rdbuf + rdl);
            cb(msg);
        }
    }
}

// ── sensor_msgs/Image → cv::Mat ──────────────────────────────────────────────
// Layout: std_msgs/Header | uint32 height | uint32 width | string encoding |
//         uint8 is_bigendian | uint32 step | uint8[] data
cv::Mat decodeImage(const BagMessage& msg) {
    const uint8_t* d = msg.data.data();
    const size_t   n = msg.data.size();

    // Skip Header: seq(4) + stamp(8) + string frame_id
    if (n < 16) return {};
    uint32_t frame_len = u32(d + 12);
    size_t pos = 16 + frame_len;

    if (pos + 13 > n) return {};
    uint32_t height = u32(d + pos);       pos += 4;
    uint32_t width  = u32(d + pos);       pos += 4;
    uint32_t enc_len = u32(d + pos);      pos += 4;
    if (pos + enc_len > n) return {};
    std::string encoding(reinterpret_cast<const char*>(d + pos), enc_len);
    pos += enc_len;
    pos += 1; // is_bigendian
    pos += 4; // step

    if (pos > n) return {};
    const uint8_t* pixels = d + pos + 4; // skip data array length prefix

    if (encoding == "mono8") {
        cv::Mat img(height, width, CV_8UC1, const_cast<uint8_t*>(pixels));
        return img.clone();
    }
    if (encoding == "bgr8") {
        cv::Mat img(height, width, CV_8UC3, const_cast<uint8_t*>(pixels));
        return img.clone();
    }
    if (encoding == "rgb8") {
        cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t*>(pixels));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }
    return {};
}
