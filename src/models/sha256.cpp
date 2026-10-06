#include "guild/models/sha256.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace guild::models {

namespace {

inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t sig0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline uint32_t sig1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline uint32_t theta0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline uint32_t theta1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

} // namespace

Sha256::Sha256() {
    reset();
}

void Sha256::reset() {
    state_[0] = 0x6a09e667;
    state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372;
    state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f;
    state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab;
    state_[7] = 0x5be0cd19;
    count_ = 0;
}

void Sha256::transform(const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               (static_cast<uint32_t>(block[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        w[i] = theta1(w[i - 2]) + w[i - 7] + theta0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + sig1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = sig0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t buf_idx = static_cast<size_t>(count_ % 64);
    count_ += len;

    if (buf_idx > 0) {
        size_t space = 64 - buf_idx;
        if (len < space) {
            std::memcpy(buffer_ + buf_idx, p, len);
            return;
        }
        std::memcpy(buffer_ + buf_idx, p, space);
        transform(buffer_);
        p += space;
        len -= space;
    }

    while (len >= 64) {
        transform(p);
        p += 64;
        len -= 64;
    }

    if (len > 0) {
        std::memcpy(buffer_, p, len);
    }
}

void Sha256::update(const std::string& str) {
    update(str.data(), str.size());
}

std::string Sha256::finalize_hex() {
    size_t buf_idx = static_cast<size_t>(count_ % 64);
    uint64_t total_bits = count_ * 8;

    buffer_[buf_idx++] = 0x80;
    if (buf_idx > 56) {
        std::memset(buffer_ + buf_idx, 0, 64 - buf_idx);
        transform(buffer_);
        buf_idx = 0;
    }
    std::memset(buffer_ + buf_idx, 0, 56 - buf_idx);
    for (int i = 0; i < 8; ++i) {
        buffer_[56 + i] = static_cast<uint8_t>((total_bits >> ((7 - i) * 8)) & 0xff);
    }
    transform(buffer_);

    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < 8; ++i) {
        ss << std::setw(8) << state_[i];
    }
    return ss.str();
}

std::string Sha256::hash_string(const std::string& str) {
    Sha256 h;
    h.update(str);
    return h.finalize_hex();
}

std::string Sha256::hash_file(const std::string& path,
                              uint64_t max_bytes,
                              std::function<void(uint64_t)> progress_cb) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return "";

    Sha256 h;
    const size_t chunk_size = 1024 * 1024; // 1 MiB chunk
    std::vector<char> buffer(chunk_size);
    uint64_t total_read = 0;

    while (file) {
        size_t to_read = chunk_size;
        if (max_bytes > 0 && total_read + to_read > max_bytes) {
            to_read = static_cast<size_t>(max_bytes - total_read);
        }
        if (to_read == 0) break;

        file.read(buffer.data(), to_read);
        std::streamsize bytes = file.gcount();
        if (bytes <= 0) break;

        h.update(buffer.data(), static_cast<size_t>(bytes));
        total_read += static_cast<uint64_t>(bytes);

        if (progress_cb) {
            progress_cb(total_read);
        }

        if (max_bytes > 0 && total_read >= max_bytes) break;
    }

    return h.finalize_hex();
}

} // namespace guild::models
