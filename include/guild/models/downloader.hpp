#pragma once

#include "guild/models/manifest.hpp"
#include "guild/models/store.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace guild::models {

struct DownloadProgress {
    std::string current_file_name;
    int current_file_index{0};
    int total_files{0};
    uint64_t file_downloaded_bytes{0};
    uint64_t file_total_bytes{0};
    uint64_t total_downloaded_bytes{0};
    uint64_t total_expected_bytes{0};
    double speed_bytes_sec{0.0};
    double percentage{0.0};
    bool is_resumed{false};
};

struct DownloadOptions {
    bool force{false};
    bool verify_hash{true};
    int max_retries{3};
    bool json_progress{false};
    std::function<bool(const DownloadProgress&)> progress_cb{nullptr};
};

class Downloader {
public:
    Downloader();
    ~Downloader();

    // Download an entire model into the local store
    bool download_model(const ModelManifest& manifest,
                        ModelStore& store,
                        const DownloadOptions& options,
                        std::string& err_msg);

    // Download a single file with resume and hash validation
    bool download_file(const std::string& url,
                       const std::string& destination_path,
                       uint64_t expected_size,
                       const std::string& expected_sha256,
                       const DownloadOptions& options,
                       DownloadProgress& progress,
                       std::string& err_msg);

    // Cancel in-flight download cleanly
    void cancel();
    bool is_cancelled() const { return cancel_requested_.load(); }

    // Utility formatting helpers
    static std::string format_bytes(uint64_t bytes);
    static std::string format_speed(double bytes_per_sec);
    static uint64_t get_available_disk_space(const std::string& path);

private:
    std::atomic<bool> cancel_requested_{false};
};

} // namespace guild::models
