// src/abi/ed25519.cpp - RFC 8032 Ed25519 Signatures & SHA-512 Implementation
//
// Copyright (c) Guild Inference Contributors. MIT License.
// Implements FIPS 180-4 SHA-512 and RFC 8032 pure Ed25519 (GF(2^255-19) extended coordinates,
// strict canonical S < L scalar check, and constant-time byte comparisons).
#include "guild/abi/ed25519.hpp"

#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace guild::abi::crypto {
namespace {

// =============================================================================
// 1. FIPS 180-4 SHA-512
// =============================================================================

constexpr uint64_t kSha512K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

inline uint64_t rotr64(uint64_t x, unsigned n) {
    return (x >> n) | (x << (64 - n));
}

inline uint64_t load_be64(const uint8_t* p) {
    return (static_cast<uint64_t>(p[0]) << 56) |
           (static_cast<uint64_t>(p[1]) << 48) |
           (static_cast<uint64_t>(p[2]) << 40) |
           (static_cast<uint64_t>(p[3]) << 32) |
           (static_cast<uint64_t>(p[4]) << 24) |
           (static_cast<uint64_t>(p[5]) << 16) |
           (static_cast<uint64_t>(p[6]) << 8)  |
           static_cast<uint64_t>(p[7]);
}

inline void store_be64(uint8_t* p, uint64_t v) {
    p[0] = static_cast<uint8_t>(v >> 56);
    p[1] = static_cast<uint8_t>(v >> 48);
    p[2] = static_cast<uint8_t>(v >> 40);
    p[3] = static_cast<uint8_t>(v >> 32);
    p[4] = static_cast<uint8_t>(v >> 24);
    p[5] = static_cast<uint8_t>(v >> 16);
    p[6] = static_cast<uint8_t>(v >> 8);
    p[7] = static_cast<uint8_t>(v);
}

void sha512_compress(uint64_t state[8], const uint8_t block[128]) {
    uint64_t w[80];
    for (int i = 0; i < 16; ++i) w[i] = load_be64(block + i * 8);
    for (int i = 16; i < 80; ++i) {
        const uint64_t s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        const uint64_t s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 80; ++i) {
        const uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        const uint64_t ch = (e & f) ^ (~e & g);
        const uint64_t temp1 = h + S1 + ch + kSha512K[i] + w[i];
        const uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        const uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint64_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// =============================================================================
// 2. GF(2^255 - 19) Field & Twisted Edwards Curve Arithmetic (RFC 8032)
// =============================================================================

using gf = int64_t[16];

constexpr gf kGf0 = {0};
constexpr gf kGf1 = {1};
constexpr gf kD = {
    0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070,
    0xe898, 0x7779, 0x4079, 0x8cc7, 0xfe73, 0x2b6f, 0x6cee, 0x5203
};
constexpr gf kD2 = {
    0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0,
    0xd130, 0xeef3, 0x80f2, 0x198e, 0xfce7, 0x56df, 0xd9dc, 0x2406
};
constexpr gf kX = {
    0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c,
    0xdc5c, 0xfdd6, 0xe231, 0xc0a4, 0x53fe, 0xcd6e, 0x36d3, 0x2169
};
constexpr gf kY = {
    0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666,
    0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666
};
constexpr gf kI = {
    0xa0b0, 0x4a0e, 0x1b27, 0xc4ee, 0xe478, 0xad2f, 0x1806, 0x2f43,
    0xd7a7, 0x3dfb, 0x0099, 0x2b4d, 0xdf0b, 0x4fc1, 0x2480, 0x2b83
};

constexpr uint64_t kL[32] = {
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
    0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10
};

void set25519(gf r, const gf a) {
    for (int i = 0; i < 16; ++i) r[i] = a[i];
}

void car25519(gf o) {
    for (int i = 0; i < 16; ++i) {
        o[i] += (1LL << 16);
        const int64_t c = o[i] >> 16;
        o[(i + 1) * (i < 15)] += c - 1 + 37 * (c - 1) * (i == 15);
        o[i] -= c << 16;
    }
}

void sel25519(gf p, gf q, int b) {
    const int64_t c = ~(b - 1);
    for (int i = 0; i < 16; ++i) {
        const int64_t t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

void pack25519(uint8_t* o, const gf n) {
    gf m, t;
    set25519(t, n);
    car25519(t);
    car25519(t);
    car25519(t);
    for (int j = 0; j < 2; ++j) {
        m[0] = t[0] - 0xffed;
        for (int i = 1; i < 15; ++i) {
            m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
            m[i - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        const int b = (m[15] >> 16) & 1;
        m[14] &= 0xffff;
        sel25519(t, m, 1 - b);
    }
    for (int i = 0; i < 16; ++i) {
        o[2 * i] = static_cast<uint8_t>(t[i] & 0xff);
        o[2 * i + 1] = static_cast<uint8_t>(t[i] >> 8);
    }
}

int neq25519(const gf a, const gf b) {
    uint8_t c[32], d[32];
    pack25519(c, a);
    pack25519(d, b);
    uint32_t diff = 0;
    for (int i = 0; i < 32; ++i) diff |= (c[i] ^ d[i]);
    return diff != 0;
}

uint8_t par25519(const gf a) {
    uint8_t d[32];
    pack25519(d, a);
    return d[0] & 1;
}

void unpack25519(gf o, const uint8_t* n) {
    for (int i = 0; i < 16; ++i) o[i] = n[2 * i] + (static_cast<int64_t>(n[2 * i + 1]) << 8);
    o[15] &= 0x7fff;
}

void A(gf o, const gf a, const gf b) {
    for (int i = 0; i < 16; ++i) o[i] = a[i] + b[i];
}

void Z(gf o, const gf a, const gf b) {
    for (int i = 0; i < 16; ++i) o[i] = a[i] - b[i];
}

void M(gf o, const gf a, const gf b) {
    int64_t t[31] = {0};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) t[i + j] += a[i] * b[j];
    }
    for (int i = 0; i < 15; ++i) t[i] += 38 * t[i + 16];
    for (int i = 0; i < 16; ++i) o[i] = t[i];
    car25519(o);
    car25519(o);
}

void S(gf o, const gf a) {
    M(o, a, a);
}

void inv25519(gf o, const gf i) {
    gf c;
    set25519(c, i);
    for (int a = 253; a >= 0; --a) {
        S(c, c);
        if (a != 2 && a != 4) M(c, c, i);
    }
    set25519(o, c);
}

void pow2523(gf o, const gf i) {
    gf c;
    set25519(c, i);
    for (int a = 250; a >= 0; --a) {
        S(c, c);
        if (a != 1) M(c, c, i);
    }
    set25519(o, c);
}

void add_pt(gf p[4], const gf q[4]) {
    gf a, b, c, d, t, e, f, g, h;
    Z(a, p[1], p[0]);
    Z(t, q[1], q[0]);
    M(a, a, t);
    A(b, p[0], p[1]);
    A(t, q[0], q[1]);
    M(b, b, t);
    M(c, p[3], q[3]);
    M(c, c, kD2);
    M(d, p[2], q[2]);
    A(d, d, d);
    Z(e, b, a);
    Z(f, d, c);
    A(g, d, c);
    A(h, b, a);

    M(p[0], e, f);
    M(p[1], h, g);
    M(p[2], g, f);
    M(p[3], e, h);
}

void cswap(gf p[4], gf q[4], uint8_t b) {
    for (int i = 0; i < 4; ++i) sel25519(p[i], q[i], b);
}

void pack_pt(uint8_t* r, gf p[4]) {
    gf tx, ty, zi;
    inv25519(zi, p[2]);
    M(tx, p[0], zi);
    M(ty, p[1], zi);
    pack25519(r, ty);
    r[31] ^= par25519(tx) << 7;
}

void scalarmult(gf p[4], gf q[4], const uint8_t* s) {
    set25519(p[0], kGf0);
    set25519(p[1], kGf1);
    set25519(p[2], kGf1);
    set25519(p[3], kGf0);
    for (int i = 255; i >= 0; --i) {
        const uint8_t b = (s[i / 8] >> (i & 7)) & 1;
        cswap(p, q, b);
        add_pt(q, p);
        add_pt(p, p);
        cswap(p, q, b);
    }
}

void scalarbase(gf p[4], const uint8_t* s) {
    gf q[4];
    set25519(q[0], kX);
    set25519(q[1], kY);
    set25519(q[2], kGf1);
    M(q[3], kX, kY);
    scalarmult(p, q, s);
}

int unpackneg(gf r[4], const uint8_t p[32]) {
    gf t, chk, num, den, den2, den4, den6;
    set25519(r[2], kGf1);
    unpack25519(r[1], p);
    S(num, r[1]);
    M(den, num, kD);
    Z(num, num, r[2]);
    A(den, r[2], den);

    S(den2, den);
    S(den4, den2);
    M(den6, den4, den2);
    M(t, den6, num);
    M(t, t, den);

    pow2523(t, t);
    M(t, t, num);
    M(t, t, den);
    M(t, t, den);
    M(r[0], t, den);

    S(chk, r[0]);
    M(chk, chk, den);
    if (neq25519(chk, num)) M(r[0], r[0], kI);

    S(chk, r[0]);
    M(chk, chk, den);
    if (neq25519(chk, num)) return -1;

    if (par25519(r[0]) == (p[31] >> 7)) Z(r[0], kGf0, r[0]);

    M(r[3], r[0], r[1]);
    return 0;
}

void modL(uint8_t* r, int64_t x[64]) {
    for (int i = 63; i >= 32; --i) {
        int64_t carry = 0;
        for (int j = i - 32; j < i - 12; ++j) {
            x[j] += carry - 16 * x[i] * static_cast<int64_t>(kL[j - (i - 32)]);
            carry = (x[j] + 128) >> 8;
            x[j] -= carry << 8;
        }
        x[i - 12] += carry;
        x[i] = 0;
    }
    int64_t carry = 0;
    for (int j = 0; j < 32; ++j) {
        x[j] += carry - (x[31] >> 4) * static_cast<int64_t>(kL[j]);
        carry = x[j] >> 8;
        x[j] &= 255;
    }
    for (int j = 0; j < 32; ++j) x[j] -= carry * static_cast<int64_t>(kL[j]);
    for (int i = 0; i < 32; ++i) {
        x[i + 1] += x[i] >> 8;
        r[i] = static_cast<uint8_t>(x[i] & 255);
    }
}

void reduce(uint8_t* r) {
    int64_t x[64];
    for (int i = 0; i < 64; ++i) x[i] = r[i];
    for (int i = 0; i < 64; ++i) r[i] = 0;
    modL(r, x);
}

// Strict RFC 8032 check: S must be in [0, L)
bool scalar_is_canonical(const uint8_t s[32]) {
    uint8_t c = 0;
    uint8_t n = 1;
    for (int i = 31; i >= 0; --i) {
        const uint8_t li = static_cast<uint8_t>(kL[i]);
        c |= ((s[i] - li) >> 8) & n;
        n &= ((s[i] ^ li) - 1) >> 8;
    }
    return c != 0;
}

} // namespace

void sha512(const uint8_t* data, size_t len, uint8_t out_digest[64]) {
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    size_t offset = 0;
    while (len - offset >= 128) {
        sha512_compress(state, data + offset);
        offset += 128;
    }
    uint8_t pad[256] = {0};
    const size_t rem = len - offset;
    if (rem > 0) std::memcpy(pad, data + offset, rem);
    pad[rem] = 0x80;
    const size_t total_blocks = (rem + 1 + 16 <= 128) ? 1 : 2;
    const uint64_t bit_len_lo = static_cast<uint64_t>(len) * 8ULL;
    const uint64_t bit_len_hi = static_cast<uint64_t>(len) >> 61;
    store_be64(pad + total_blocks * 128 - 16, bit_len_hi);
    store_be64(pad + total_blocks * 128 - 8, bit_len_lo);
    sha512_compress(state, pad);
    if (total_blocks == 2) sha512_compress(state, pad + 128);
    for (int i = 0; i < 8; ++i) store_be64(out_digest + i * 8, state[i]);
}

std::string bytes_to_hex(const uint8_t* data, size_t len) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out[2 * i] = kHex[(data[i] >> 4) & 0xF];
        out[2 * i + 1] = kHex[data[i] & 0xF];
    }
    return out;
}

bool hex_to_bytes(const std::string& hex, uint8_t* out, size_t expected_bytes) {
    if (hex.size() != expected_bytes * 2) return false;
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < expected_bytes; ++i) {
        const int hi = nibble(hex[2 * i]);
        const int lo = nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

Ed25519KeyPair ed25519_keypair_from_seed(const uint8_t seed[32]) {
    Ed25519KeyPair kp{};
    std::memcpy(kp.seed.data(), seed, 32);
    uint8_t d[64];
    sha512(seed, 32, d);
    d[0] &= 248;
    d[31] &= 127;
    d[31] |= 64;

    gf p[4];
    scalarbase(p, d);
    pack_pt(kp.public_key.data(), p);

    std::memcpy(kp.secret_key.data(), seed, 32);
    std::memcpy(kp.secret_key.data() + 32, kp.public_key.data(), 32);
    kp.seed_hex = bytes_to_hex(kp.seed.data(), 32);
    kp.public_key_hex = bytes_to_hex(kp.public_key.data(), 32);
    return kp;
}

bool ed25519_keypair_from_seed_hex(const std::string& seed_hex, Ed25519KeyPair& out_kp) {
    uint8_t seed[32];
    if (!hex_to_bytes(seed_hex, seed, 32)) return false;
    out_kp = ed25519_keypair_from_seed(seed);
    return true;
}

bool ed25519_generate_keypair(Ed25519KeyPair& out_kp) {
    uint8_t seed[32] = {0};
    bool ok = false;
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (urandom.good()) {
        urandom.read(reinterpret_cast<char*>(seed), 32);
        if (urandom.gcount() == 32) ok = true;
    }
    if (!ok) {
        std::random_device rd;
        for (int i = 0; i < 32; ++i) seed[i] = static_cast<uint8_t>(rd() & 0xff);
    }
    out_kp = ed25519_keypair_from_seed(seed);
    return true;
}

void ed25519_sign(const uint8_t* msg,
                  size_t msg_len,
                  const uint8_t seed[32],
                  const uint8_t public_key[32],
                  uint8_t out_sig[64]) {
    uint8_t d[64];
    sha512(seed, 32, d);
    d[0] &= 248;
    d[31] &= 127;
    d[31] |= 64;

    // r = SHA-512(d[32..63] || msg) mod L
    std::vector<uint8_t> buf(32 + msg_len);
    std::memcpy(buf.data(), d + 32, 32);
    if (msg_len > 0) std::memcpy(buf.data() + 32, msg, msg_len);
    uint8_t r[64];
    sha512(buf.data(), buf.size(), r);
    reduce(r);

    gf p[4];
    scalarbase(p, r);
    pack_pt(out_sig, p);

    // hram = SHA-512(R || A || msg) mod L
    std::vector<uint8_t> hbuf(64 + msg_len);
    std::memcpy(hbuf.data(), out_sig, 32);
    std::memcpy(hbuf.data() + 32, public_key, 32);
    if (msg_len > 0) std::memcpy(hbuf.data() + 64, msg, msg_len);
    uint8_t hram[64];
    sha512(hbuf.data(), hbuf.size(), hram);
    reduce(hram);

    int64_t x[64] = {0};
    for (int i = 0; i < 32; ++i) x[i] = r[i];
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) x[i + j] += static_cast<int64_t>(hram[i]) * static_cast<int64_t>(d[j]);
    }
    modL(out_sig + 32, x);
}

bool ed25519_verify(const uint8_t* msg,
                    size_t msg_len,
                    const uint8_t sig[64],
                    const uint8_t public_key[32]) {
    if (!sig || !public_key) return false;
    if (!scalar_is_canonical(sig + 32)) return false;

    gf p[4], q[4];
    if (unpackneg(q, public_key) != 0) return false;

    std::vector<uint8_t> hbuf(64 + msg_len);
    std::memcpy(hbuf.data(), sig, 32);
    std::memcpy(hbuf.data() + 32, public_key, 32);
    if (msg_len > 0) std::memcpy(hbuf.data() + 64, msg, msg_len);
    uint8_t h[64];
    sha512(hbuf.data(), hbuf.size(), h);
    reduce(h);

    scalarmult(p, q, h);
    scalarbase(q, sig + 32);
    add_pt(p, q);
    uint8_t t[32];
    pack_pt(t, p);

    uint32_t diff = 0;
    for (int i = 0; i < 32; ++i) diff |= (sig[i] ^ t[i]);
    return diff == 0;
}

bool ed25519_sign_hex(const std::string& message,
                      const std::string& seed_hex,
                      std::string& out_sig_hex,
                      std::string* out_pubkey_hex) {
    Ed25519KeyPair kp;
    if (!ed25519_keypair_from_seed_hex(seed_hex, kp)) return false;
    uint8_t sig[64];
    ed25519_sign(reinterpret_cast<const uint8_t*>(message.data()), message.size(),
                 kp.seed.data(), kp.public_key.data(), sig);
    out_sig_hex = bytes_to_hex(sig, 64);
    if (out_pubkey_hex) *out_pubkey_hex = kp.public_key_hex;
    return true;
}

bool ed25519_verify_hex(const std::string& message,
                        const std::string& sig_hex,
                        const std::string& pubkey_hex) {
    uint8_t sig[64];
    uint8_t pub[32];
    if (!hex_to_bytes(sig_hex, sig, 64) || !hex_to_bytes(pubkey_hex, pub, 32)) return false;
    return ed25519_verify(reinterpret_cast<const uint8_t*>(message.data()), message.size(), sig, pub);
}

} // namespace guild::abi::crypto
