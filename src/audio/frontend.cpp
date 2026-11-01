// src/audio/frontend.cpp - audio in, log-mel features out (include/guild/audio/frontend.hpp).
//
// Built with -ffp-contract=off: each single-precision product and sum rounds on its own, as the reference's float32
// tensor operations do (an FMA would round once and give other bits).
#include "guild/audio/frontend.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <thread>

namespace guild::audio {

namespace {

uint32_t le32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint16_t le16(const uint8_t* p) {
    return uint16_t(p[0] | p[1] << 8);
}

/// I0, the modified Bessel function of the first kind (its power series; |x| <= ~30 here)
double bessel_i0(double x) {
    double sum = 1.0, term = 1.0, q = x * x / 4.0;
    for (int k = 1; k < 200; ++k) {
        term *= q / (double(k) * double(k));
        sum += term;
        if (term < sum * 1e-17)
            break;
    }
    return sum;
}

/// The resampler's low-pass at offset d input samples: a sinc at `cut` cycles a sample, Kaiser-windowed to +-half
double lowpass(double d, double cut, double half, double beta, double i0_beta) {
    if (std::fabs(d) >= half)
        return 0.0;
    double x = 2.0 * cut * d;
    double s = x == 0.0 ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
    double u = d / half;
    return 2.0 * cut * s * bessel_i0(beta * std::sqrt(1.0 - u * u)) / i0_beta;
}

/// An in-place radix-2 FFT of n (a power of two) complex values, in double
void fft(std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& tw) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                std::complex<double> w = tw[k * step];
                std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
    }
}

} // namespace

bool read_wav(const std::string& path, Wave& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(b.data() + 8, "WAVE", 4) != 0) {
        err = path + ": not a RIFF/WAVE file";
        return false;
    }
    int fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data = nullptr;
    size_t data_bytes = 0;
    for (size_t p = 12; p + 8 <= b.size();) {
        uint32_t len = le32(b.data() + p + 4);
        const uint8_t* body = b.data() + p + 8;
        size_t avail = std::min<size_t>(len, b.size() - p - 8);
        if (std::memcmp(b.data() + p, "fmt ", 4) == 0 && avail >= 16) {
            fmt = le16(body);
            channels = le16(body + 2);
            rate = le32(body + 4);
            bits = le16(body + 14);
            if (fmt == 0xFFFE && avail >= 26) // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes
                fmt = le16(body + 24);
        } else if (std::memcmp(b.data() + p, "data", 4) == 0) {
            data = body;
            data_bytes = avail;
        }
        p += 8 + size_t(len) + (len & 1);
    }
    if (!data || channels <= 0 || rate == 0) {
        err = path + ": no fmt or data chunk";
        return false;
    }
    const bool is_float = fmt == 3, is_pcm = fmt == 1;
    if (!(is_pcm && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) &&
        !(is_float && (bits == 32 || bits == 64))) {
        err = path + ": unsupported WAV format " + std::to_string(fmt) + " with " + std::to_string(bits) + " bits";
        return false;
    }
    const int bytes = bits / 8;
    const size_t frames = data_bytes / (size_t(bytes) * size_t(channels));
    out.rate = int(rate);
    out.samples.assign(frames, 0.f);
    // libsndfile's conversions to float: integers scaled by 1 / 2^(bits - 1), 8-bit unsigned centered first
    auto sample = [&](const uint8_t* s) -> float {
        if (is_float) {
            if (bits == 32) {
                float v;
                std::memcpy(&v, s, 4);
                return v;
            }
            double v;
            std::memcpy(&v, s, 8);
            return float(v);
        }
        switch (bits) {
        case 8: return float(int(s[0]) - 128) * (1.0f / 128.0f);
        case 16: return float(int16_t(le16(s))) * (1.0f / 32768.0f);
        case 24: {
            int32_t v = int32_t(uint32_t(s[0]) << 8 | uint32_t(s[1]) << 16 | uint32_t(s[2]) << 24) >> 8;
            return float(v) * (1.0f / 8388608.0f);
        }
        default: return float(double(int32_t(le32(s))) * (1.0 / 2147483648.0));
        }
    };
    for (size_t i = 0; i < frames; ++i) {
        const uint8_t* s = data + i * size_t(bytes) * size_t(channels);
        if (channels == 1) {
            out.samples[i] = sample(s);
            continue;
        }
        float acc = 0.f; // numpy's float32 mean over the channels: the float32 sum, then / channels
        for (int c = 0; c < channels; ++c)
            acc += sample(s + size_t(c) * size_t(bytes));
        out.samples[i] = acc / float(channels);
    }
    return true;
}

bool write_wav_pcm16(const std::string& path, const std::vector<float>& samples, int rate, std::string& err) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot write " + path;
        return false;
    }
    auto u32 = [&](uint32_t v) {
        uint8_t c[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
        f.write(reinterpret_cast<const char*>(c), 4);
    };
    auto u16 = [&](uint16_t v) {
        uint8_t c[2] = {uint8_t(v), uint8_t(v >> 8)};
        f.write(reinterpret_cast<const char*>(c), 2);
    };
    const uint32_t data_bytes = uint32_t(samples.size() * 2);
    f.write("RIFF", 4);
    u32(36 + data_bytes);
    f.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(uint32_t(rate));
    u32(uint32_t(rate) * 2);
    u16(2);
    u16(16);
    f.write("data", 4);
    u32(data_bytes);
    for (float s : samples) {
        long v = std::lrint(double(s) * 32768.0);
        u16(uint16_t(int16_t(std::clamp(v, -32768L, 32767L))));
    }
    return bool(f);
}

std::vector<float> resample(const std::vector<float>& x, int from, int to) {
    if (from == to || x.empty())
        return x;
    const int64_t g = std::gcd(int64_t(from), int64_t(to));
    const int64_t L = to / g, M = from / g;                            // output sample n sits at input time n * M / L
    const double fc = 0.5 * double(std::min(from, to)) / double(from); // the lower Nyquist, cycles an input sample
    const double atten = 125.0;
    const double beta = 0.1102 * (atten - 8.7);
    const double width = (1.0 - 0.913) * fc; // the transition band, 0.913 .. 1.0 of the lower Nyquist
    const double half = (atten - 7.95) / (2.285 * 2.0 * M_PI * width) / 2.0;
    const double cut = 0.5 * (0.913 + 1.0) * fc;
    const double i0_beta = bessel_i0(beta);
    const int64_t H = int64_t(std::ceil(half));
    const int64_t n_in = int64_t(x.size());
    const int64_t n_out = (n_in * L + M - 1) / M;
    std::vector<float> y(size_t(n_out), 0.f);
    // taps of phase p (fractional offset p / L): input samples k0 - H + 1 .. k0 + H
    const bool table = L <= 8192;
    std::vector<double> taps;
    if (table) {
        taps.resize(size_t(L * 2 * H));
        for (int64_t p = 0; p < L; ++p)
            for (int64_t j = 0; j < 2 * H; ++j)
                taps[size_t(p * 2 * H + j)] =
                    lowpass(double(p) / double(L) + double(H - 1 - j), cut, half, beta, i0_beta);
    }
    for (int64_t n = 0; n < n_out; ++n) {
        const int64_t num = n * M, k0 = num / L, p = num % L;
        double acc = 0.0;
        for (int64_t j = 0; j < 2 * H; ++j) {
            const int64_t k = k0 - H + 1 + j;
            if (k < 0 || k >= n_in)
                continue;
            const double w = table ? taps[size_t(p * 2 * H + j)]
                                   : lowpass(double(p) / double(L) + double(H - 1 - j), cut, half, beta, i0_beta);
            acc += w * double(x[size_t(k)]);
        }
        y[size_t(n)] = float(acc);
    }
    return y;
}

bool log_mel(const float* x, int64_t n, const MelConfig& cfg, Features& out, std::string& err) {
    const int bins = cfg.n_fft / 2 + 1;
    if (int(cfg.window.size()) != cfg.win || int(cfg.filters.size()) != cfg.n_mels * bins || cfg.win > cfg.n_fft ||
        (cfg.n_fft & (cfg.n_fft - 1)) != 0) {
        err = "log_mel: the window or the filter bank does not match the configuration";
        return false;
    }
    // pre-emphasis: x[0], x[i] - 0.97 x[i - 1] (a float32 product, then a float32 difference)
    std::vector<float> e((size_t) n);
    for (int64_t i = 0; i < n; ++i)
        e[size_t(i)] = i == 0 ? x[0] : x[i] - cfg.preemphasis * x[i - 1];
    // a centered STFT: n_fft / 2 zeros each side, frame t at t * hop, the window centered in n_fft
    const int pad = cfg.n_fft / 2, woff = (cfg.n_fft - cfg.win) / 2;
    const int64_t n_frames = 1 + n / cfg.hop; // torch.stft's count; the last one is past the valid frames
    out.frames = int(n / cfg.hop);
    out.n_mels = cfg.n_mels;
    // the reference pads the centered STFT's frame count (one more than the valid frames: the last one is masked) to
    // whole groups, so frames that fill whole groups get one group of padding more, which the audio model masks
    out.padded = (int(n_frames) + cfg.stack - 1) / cfg.stack * cfg.stack;
    out.mel.assign(size_t(out.padded) * size_t(cfg.n_mels), 0.f);
    std::vector<std::complex<double>> tw(size_t(cfg.n_fft / 2));
    for (int k = 0; k < cfg.n_fft / 2; ++k)
        tw[size_t(k)] = std::polar(1.0, -2.0 * M_PI * double(k) / double(cfg.n_fft));
    // frames are independent: a few threads take every n-th frame (each its own FFT buffer)
    auto frames = [&](int first, int step) {
        std::vector<std::complex<double>> a((size_t) cfg.n_fft);
        std::vector<float> power((size_t) bins);
        for (int64_t t = first; t < std::min<int64_t>(n_frames, out.frames); t += step) {
            for (int i = 0; i < cfg.n_fft; ++i) {
                const int64_t s = t * cfg.hop + i - pad;
                const int w = i - woff;
                float v = 0.f;
                if (s >= 0 && s < n && w >= 0 && w < cfg.win)
                    v = e[size_t(s)] * cfg.window[size_t(w)];
                a[size_t(i)] = {double(v), 0.0};
            }
            fft(a, tw);
            for (int k = 0; k < bins; ++k) {
                // |z| as torch computes it: sqrt(re^2 + im^2) in float32, then squared
                const float re = float(a[size_t(k)].real()), im = float(a[size_t(k)].imag());
                const float re2 = re * re, im2 = im * im;
                const float mag = std::sqrt(re2 + im2);
                power[size_t(k)] = mag * mag;
            }
            float* row = out.mel.data() + size_t(t) * size_t(cfg.n_mels);
            for (int m = 0; m < cfg.n_mels; ++m) {
                const float* fb = cfg.filters.data() + size_t(m) * size_t(bins);
                double acc = 0.0;
                for (int k = 0; k < bins; ++k)
                    acc += double(fb[k]) * double(power[size_t(k)]);
                row[m] = float(std::log(double(float(acc) + cfg.guard)));
            }
        }
    };
    const int nt =
        (int) std::clamp<int64_t>(out.frames / 64, 1, std::min(16u, std::max(1u, std::thread::hardware_concurrency())));
    std::vector<std::thread> th;
    for (int i = 1; i < nt; ++i)
        th.emplace_back(frames, i, nt);
    frames(0, nt);
    for (auto& x : th)
        x.join();
    return true;
}

namespace {

/// PyTorch's float32 sum over the rows of a column (ATen's multi_row_sum, cpu/SumKernel.cpp): four levels of
/// accumulators, 2^max(4, ceil(log2 n) / 4) rows a step, every addition rounded to float.  A mel bin that hardly
/// varies over the clip is divided by a standard deviation near 1e-5, so the mean has to be the reference's to the
/// last bit (one unit in the last place of a mean near -16 is 0.08 after the division).
float torch_column_sum(const float* x, int64_t stride, int64_t n) {
    constexpr int kLevels = 4;
    int64_t ceil_log2 = 0;
    while ((int64_t(1) << ceil_log2) < n)
        ++ceil_log2;
    const int64_t power = std::max<int64_t>(4, ceil_log2 / kLevels), step = int64_t(1) << power, mask = step - 1;
    float acc[kLevels] = {0.f, 0.f, 0.f, 0.f};
    int64_t i = 0;
    for (; i + step <= n;) {
        for (int64_t j = 0; j < step; ++j, ++i)
            acc[0] += x[i * stride];
        for (int j = 1; j < kLevels; ++j) {
            acc[j] += acc[j - 1];
            acc[j - 1] = 0.f;
            if ((i & (mask << (j * power))) != 0)
                break;
        }
    }
    for (; i < n; ++i)
        acc[0] += x[i * stride];
    for (int j = 1; j < kLevels; ++j)
        acc[0] += acc[j];
    return acc[0];
}

} // namespace

std::vector<float> normalize_per_feature(const Features& f) {
    // the reference's float32 steps over every frame held (zeros past the valid ones): sum, / n, centered, squared,
    // sum, / (n - 1), sqrt, + 1e-5, divided
    const int M = f.n_mels, T = f.frames, P = f.padded;
    std::vector<float> out(f.mel.size(), 0.f), c((size_t) P);
    for (int m = 0; m < M; ++m) {
        const float* col = f.mel.data() + m;
        const float mean = torch_column_sum(col, M, P) / float(std::max(T, 1));
        for (int t = 0; t < P; ++t)
            c[(size_t) t] = t < T ? col[(size_t) t * (size_t) M] - mean : 0.f;
        for (int t = 0; t < P; ++t)
            c[(size_t) t] = c[(size_t) t] * c[(size_t) t];
        const float var = torch_column_sum(c.data(), 1, P) / float(std::max(T - 1, 1));
        const float sd = std::sqrt(var) + 1e-5f;
        for (int t = 0; t < T; ++t)
            out[(size_t) t * (size_t) M + (size_t) m] = (col[(size_t) t * (size_t) M] - mean) / sd;
    }
    return out;
}

} // namespace guild::audio
