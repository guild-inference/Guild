#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace guild::models {

class Sha256 {
public:
    Sha256();
    void reset();
    void update(const void* data, size_t len);
    void update(const std::string& str);
    std::string finalize_hex();

    static std::string hash_string(const std::string& str);
    static std::string hash_file(const std::string& path,
                                 uint64_t max_bytes = 0,
                                 std::function<void(uint64_t bytes_hashed)> progress_cb = nullptr);

private:
    uint32_t state_[8];
    uint64_t count_{0};
    uint8_t buffer_[64];

    void transform(const uint8_t block[64]);
};

} // namespace guild::models
