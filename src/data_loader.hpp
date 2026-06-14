#pragma once

#include "../cpp/bag_reader.hpp"
#include <atomic>
#include <chrono>
#include <thread>

// Replays a ROS1 bag at a controllable rate.
// Usage:
//   DataLoader streamer("file.bag", 1.0);      // real-time
//   streamer.addCallback("/imu0", cb);
//   streamer.play();   // blocks until end-of-bag or stop() is called
class DataLoader {
public:
    explicit DataLoader(const std::string& path, double speed = 1.0)
        : path_(path), speed_(speed) {}

    void addCallback(const std::string& topic, MessageCallback cb) {
        callbacks_[topic].push_back(std::move(cb));
    }

    // Block until all messages have been delivered or stop() is called.
    void play() {
        stopped_ = false;
        BagReader reader(path_);

        std::vector<std::string> topics;
        for (auto& [t, _] : callbacks_) topics.push_back(t);

        double last_bag_time   = -1.0;
        auto   last_wall_clock = std::chrono::steady_clock::now();

        reader.readMessages([&](const BagMessage& msg) {
            if (stopped_) return;

            // Pace delivery to match original timing scaled by speed_
            double t = msg.stamp.toSec();
            if (last_bag_time > 0.0) {
                double bag_dt  = (t - last_bag_time) / speed_;
                auto   elapsed = std::chrono::steady_clock::now() - last_wall_clock;
                auto   sleep   = std::chrono::duration<double>(bag_dt) - elapsed;
                if (sleep.count() > 0)
                    std::this_thread::sleep_for(sleep);
            }
            last_bag_time   = t;
            last_wall_clock = std::chrono::steady_clock::now();

            auto it = callbacks_.find(msg.topic);
            if (it != callbacks_.end())
                for (auto& cb : it->second) cb(msg);
        }, topics);
    }

    void stop() { stopped_ = true; }

private:
    std::string  path_;
    double       speed_;
    std::atomic<bool> stopped_{false};
    std::map<std::string, std::vector<MessageCallback>> callbacks_;
};
